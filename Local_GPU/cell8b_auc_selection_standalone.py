# ===================================================================
# Cell 8B: Full 5-Fold Training — AUC-Based Checkpoint Selection (ALTERNATE)
# ===================================================================
# Same model/data/scheduler as Cell 8 (OneCycleLR, batch 16, AMP,
# patience 5) — the ONLY difference is what decides "best epoch" and
# when to stop:
#   Cell 8  (loss-based): checkpoint saved when vl_loss improves,
#                          patience counted on vl_loss plateaus.
#   Cell 8B (AUC-based):  checkpoint saved when vl_auc improves,
#                          patience counted on vl_auc plateaus.
#
# WHY: across Folds 1-4 of the Cell 8 run, the loss-selected epoch and
# the epoch with the actual peak val AUC disagreed in 3/4 folds
# (Fold 1: 0.8821 vs peak 0.9180, Fold 2: 0.8591 vs peak 0.9035,
#  Fold 3: 0.8125 vs peak 0.8249) — only Fold 4 agreed, and that was
# the one fold with a smooth, non-early-stopped trajectory. This run
# selects directly on AUC to see whether that gap is real or an
# artifact of loss-based selection, for a clean two-run comparison.
#
# Writes to SEPARATE checkpoint files (`*_aucsel.pth`) so Cell 8's
# loss-selected checkpoints are never touched — both runs' results
# can be compared side by side afterwards.
#
# RESUMABLE the same way Cell 8 is: safe to stop and re-run; skips
# completed folds and resumes an interrupted fold from its last saved
# epoch.
# ===================================================================
import gc

USE_CACHE  = True
FAST_PROTO = False  # ← KEEP FALSE for thesis results

if not IS_GPU:
    print("⚠ No GPU detected. This cell takes 69–129 hrs on CPU.")
    _c = input("Continue? (yes/no): ")
    if _c.lower() != 'yes': raise SystemExit("Cancelled — use CPU notebook instead.")

if FAST_PROTO:
    print("FAST_PROTO mode: 3 folds, 10 epochs, 40% data")
    _df_train = df.groupby('diagnosis', group_keys=False).apply(
        lambda x: x.sample(frac=0.4, random_state=42)).reset_index(drop=True)
    _n_splits, _n_epochs = 3, 10
else:
    _df_train = df.copy().reset_index(drop=True)
    _n_splits, _n_epochs = N_SPLITS, NUM_EPOCHS

print(f"Training (AUC-selected): {len(_df_train)} scans | {_n_splits} folds × {_n_epochs} epochs")
print(f"GPU:{IS_GPU} | Batch:{BATCH_SIZE} | AMP:{USE_AMP} | Workers:{NUM_WORKERS}")

X_indices = np.arange(len(_df_train)); y_labels = _df_train['label_encoded'].values
skf = StratifiedKFold(n_splits=_n_splits, shuffle=True, random_state=42)
fold_results_auc = []; best_val_auc_overall = -1.0; best_fold_idx_overall = 0

with mlflow.start_run(run_name="NeuroDT-GPU-5Fold-AUCSelect") as run:
    mlflow.log_params({'num_epochs':_n_epochs,'lr':LEARNING_RATE,'batch_size':BATCH_SIZE,
                       'n_splits':_n_splits,'use_amp':USE_AMP,'scheduler':'OneCycleLR',
                       'patience':EARLY_STOPPING_PATIENCE,'selection_metric':'auc'})

    for fold_idx,(train_idx,val_idx) in enumerate(skf.split(X_indices,y_labels)):
        print(f"\n{'='*55}\n  FOLD {fold_idx+1} / {_n_splits}  (AUC-selected)\n{'='*55}")
        ckpt_path=BEST_MODEL_DIR/f'best_model_fold{fold_idx+1}_aucsel.pth'

        # ── Resume check: is this fold already done? ─────────────────────
        # weights_only=False: these checkpoints intentionally embed a fitted
        # StandardScaler + label_map alongside the tensors — that's our own
        # trusted data, not an untrusted external file.
        resume_ckpt = None
        if ckpt_path.exists():
            _existing = torch.load(ckpt_path, map_location=device, weights_only=False)
            _cfg = _existing.get('training_config', {})
            _is_this_run = _cfg.get('scheduler') == 'OneCycleLR' and _cfg.get('selection_metric') == 'auc'
            if not _is_this_run:
                print(f"  ⚠ {ckpt_path.name} exists but is from a different run "
                      f"— starting Fold {fold_idx+1} fresh; it will be overwritten.")
            elif _existing.get('fold_complete'):
                print(f"  ✓ Fold {fold_idx+1} already complete "
                      f"(epoch {_existing['epoch']}, AUC={_existing['val_auc']:.4f}) — skipping.")
                fold_results_auc.append({'fold':fold_idx+1,'best_val_loss':_existing['val_loss'],
                                      'best_val_auc':_existing['val_auc'],'ckpt_path':str(ckpt_path),
                                      'scaler':_existing['scaler']})
                if _existing['val_auc']>best_val_auc_overall: best_val_auc_overall=_existing['val_auc']; best_fold_idx_overall=fold_idx
                continue
            else:
                resume_ckpt = _existing
                print(f"  ↻ Resuming Fold {fold_idx+1} from epoch {_existing['epoch']+1} "
                      f"(last val_auc={_existing['val_auc']:.4f}, loss={_existing['val_loss']:.4f})")

        df_tr=_df_train.iloc[train_idx]; df_vl=_df_train.iloc[val_idx]
        fold_scaler = resume_ckpt['scaler'] if resume_ckpt else StandardScaler().fit(df_tr[TABULAR_FEATURES].values)

        train_ds=CachedAdniDataset(df_tr,CACHE_DIR,TABULAR_FEATURES,scaler=fold_scaler,
            fallback_transforms=train_transforms,mount_path=LOCAL_MOUNT_PATH)
        val_ds=CachedAdniDataset(df_vl,CACHE_DIR,TABULAR_FEATURES,scaler=fold_scaler,
            fallback_transforms=val_transforms,mount_path=None)
        train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,
            num_workers=NUM_WORKERS,pin_memory=IS_GPU,persistent_workers=IS_GPU and NUM_WORKERS>0)
        val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,
            num_workers=NUM_WORKERS,pin_memory=IS_GPU)

        fold_model=MultimodalTransformer(tabular_dim=len(TABULAR_FEATURES),
                                          num_classes=len(le.classes_)).to(device)
        optimizer=torch.optim.AdamW(fold_model.parameters(),lr=LEARNING_RATE,weight_decay=1e-4)
        # OneCycleLR: 10% warmup — identical schedule to Cell 8
        scheduler=torch.optim.lr_scheduler.OneCycleLR(
            optimizer,max_lr=LEARNING_RATE,steps_per_epoch=len(train_loader),
            epochs=_n_epochs,pct_start=0.10,anneal_strategy='cos',
            div_factor=10.0,final_div_factor=1e4)
        criterion=nn.CrossEntropyLoss(weight=class_weights_tensor)
        scaler_amp=torch.cuda.amp.GradScaler(enabled=USE_AMP)
        best_val_auc_fold=-1.0; patience_count=0; start_epoch=1

        if resume_ckpt is not None:
            fold_model.load_state_dict(resume_ckpt['model_state_dict'])
            if 'optimizer_state_dict' in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in resume_ckpt:
                scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
            if 'scaler_amp_state_dict' in resume_ckpt:
                scaler_amp.load_state_dict(resume_ckpt['scaler_amp_state_dict'])
            best_val_auc_fold=resume_ckpt['val_auc']
            start_epoch=resume_ckpt['epoch']+1
            if start_epoch>_n_epochs:
                print(f"  Fold {fold_idx+1} checkpoint is already past the last epoch — treating as complete.")
                resume_ckpt['fold_complete']=True
                torch.save(resume_ckpt, ckpt_path)
                fold_results_auc.append({'fold':fold_idx+1,'best_val_loss':resume_ckpt['val_loss'],
                                      'best_val_auc':resume_ckpt['val_auc'],'ckpt_path':str(ckpt_path),
                                      'scaler':resume_ckpt['scaler']})
                if resume_ckpt['val_auc']>best_val_auc_overall: best_val_auc_overall=resume_ckpt['val_auc']; best_fold_idx_overall=fold_idx
                continue

        for epoch in range(start_epoch,_n_epochs+1):
            fold_model.train(); tr_loss=0.0; tr_preds=[]; tr_labels=[]
            for imgs,tabs,labels in tqdm(train_loader,desc=f'  Ep {epoch:02d} train',leave=False):
                imgs,tabs,labels=imgs.to(device),tabs.to(device),labels.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    logits=fold_model(imgs,tabs); loss=criterion(logits,labels)
                scaler_amp.scale(loss).backward(); scaler_amp.step(optimizer)
                scaler_amp.update(); scheduler.step()
                tr_loss+=loss.item(); tr_preds+=logits.argmax(1).cpu().tolist()
                tr_labels+=labels.cpu().tolist()
            tr_loss/=len(train_loader); tr_acc=accuracy_score(tr_labels,tr_preds)

            fold_model.eval(); vl_loss=0.0; vl_preds=[]; vl_labels=[]; vl_probs=[]
            with torch.no_grad():
                for imgs,tabs,labels in tqdm(val_loader,desc=f'  Ep {epoch:02d} val  ',leave=False):
                    imgs,tabs,labels=imgs.to(device),tabs.to(device),labels.to(device)
                    with torch.cuda.amp.autocast(enabled=USE_AMP):
                        logits=fold_model(imgs,tabs); loss=criterion(logits,labels)
                    vl_loss+=loss.item()
                    vl_probs+=torch.softmax(logits.float(),1).cpu().tolist()
                    vl_preds+=logits.argmax(1).cpu().tolist()
                    vl_labels+=labels.cpu().tolist()
            vl_loss/=len(val_loader)
            vl_acc=accuracy_score(vl_labels,vl_preds)
            vl_bal=balanced_accuracy_score(vl_labels,vl_preds)
            vl_auc=roc_auc_score(vl_labels,vl_probs,multi_class='ovr',average='macro',
                                   labels=np.unique(y_labels))
            lr_now=optimizer.param_groups[0]['lr']
            print(f'  Ep {epoch:02d} | lr={lr_now:.2e} | tr_loss={tr_loss:.4f} acc={tr_acc:.3f} | '
                  f'vl_loss={vl_loss:.4f} acc={vl_acc:.3f} bal={vl_bal:.3f} AUC={vl_auc:.4f}')
            mlflow.log_metrics({f'f{fold_idx+1}_auc_vl_loss':vl_loss,f'f{fold_idx+1}_auc_vl_auc':vl_auc,
                                 f'f{fold_idx+1}_auc_tr_loss':tr_loss},step=fold_idx*_n_epochs+epoch)

            if vl_auc>best_val_auc_fold:
                best_val_auc_fold=vl_auc; patience_count=0
                torch.save({'fold':fold_idx+1,'epoch':epoch,'model_state_dict':fold_model.state_dict(),
                            'val_loss':vl_loss,'val_auc':vl_auc,'scaler':fold_scaler,'label_map':label_map,
                            'optimizer_state_dict':optimizer.state_dict(),
                            'scheduler_state_dict':scheduler.state_dict(),
                            'scaler_amp_state_dict':scaler_amp.state_dict(),
                            'fold_complete':False,
                            'training_config':{'scheduler':'OneCycleLR','batch_size':BATCH_SIZE,
                                               'use_amp':USE_AMP,'selection_metric':'auc'}},ckpt_path)
            else:
                patience_count+=1
                if patience_count>=EARLY_STOPPING_PATIENCE:
                    print(f'  Early stopping at epoch {epoch}'); break

        ckpt=torch.load(ckpt_path,map_location=device,weights_only=False)
        ckpt['fold_complete']=True
        torch.save(ckpt, ckpt_path)
        fold_results_auc.append({'fold':fold_idx+1,'best_val_loss':ckpt['val_loss'],
                              'best_val_auc':ckpt['val_auc'],'ckpt_path':str(ckpt_path),'scaler':fold_scaler})
        if ckpt['val_auc']>best_val_auc_overall: best_val_auc_overall=ckpt['val_auc']; best_fold_idx_overall=fold_idx
        print(f"  Fold {fold_idx+1} best: AUC={ckpt['val_auc']:.4f}  (vl_loss at that epoch={ckpt['val_loss']:.4f})")
        gc.collect()
        if IS_GPU: torch.cuda.empty_cache()

aucs_auc=[r['best_val_auc'] for r in fold_results_auc]
print(f"\n{'='*55}")
print(f"  5-FOLD CV SUMMARY (GPU, AUC-selected)")
print(f"  Mean AUC: {np.mean(aucs_auc):.4f} ± {np.std(aucs_auc):.4f}")
print(f"  Best fold: {best_fold_idx_overall+1}  (AUC={best_val_auc_overall:.4f})")
for r in fold_results_auc: print(f"  Fold {r['fold']}: AUC={r['best_val_auc']:.4f}")
print(f"{'='*55}")
mlflow.log_metrics({'cv_mean_auc_aucsel':np.mean(aucs_auc),'cv_std_auc_aucsel':np.std(aucs_auc)})

# ── Side-by-side vs. the loss-selected Cell 8 run, if it's still in memory ──
if 'fold_results' in dir() and fold_results:
    aucs_loss=[r['best_val_auc'] for r in fold_results]
    print(f"\n{'='*55}")
    print(f"  LOSS-SELECTED (Cell 8)  vs  AUC-SELECTED (Cell 8B)")
    print(f"{'='*55}")
    for i in range(min(len(fold_results),len(fold_results_auc))):
        l=next(r for r in fold_results if r['fold']==i+1)
        a=next(r for r in fold_results_auc if r['fold']==i+1)
        print(f"  Fold {i+1}: loss-sel={l['best_val_auc']:.4f}  |  auc-sel={a['best_val_auc']:.4f}"
              f"  (Δ={a['best_val_auc']-l['best_val_auc']:+.4f})")
    print(f"  Mean:    loss-sel={np.mean(aucs_loss):.4f} ± {np.std(aucs_loss):.4f}  |  "
          f"auc-sel={np.mean(aucs_auc):.4f} ± {np.std(aucs_auc):.4f}")
    print(f"{'='*55}")
