# ===================================================================
# Cell 8: Full 5-Fold Training (GPU VERSION)
# ===================================================================
# KEY DIFFERENCE from original Cell 8:
#   Scheduler: OneCycleLR with 10% warmup (replaces CosineAnnealingLR)
#   This fixes the epoch-1 loss spike that caused early stopping on CPU.
#   Batch 16 + AMP + patience 5 = stable training across all 5 folds.
#
# RESUMABLE: safe to stop (Ctrl+C / kernel death / closing the notebook)
# and simply re-run this cell later. On restart it:
#   - skips any fold already marked complete (loads its saved result)
#   - resumes a fold that was cut off mid-training from its last saved
#     epoch, restoring optimizer/scheduler/AMP-scaler state so training
#     continues smoothly rather than restarting that fold from epoch 1
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

print(f"Training: {len(_df_train)} scans | {_n_splits} folds × {_n_epochs} epochs")
print(f"GPU:{IS_GPU} | Batch:{BATCH_SIZE} | AMP:{USE_AMP} | Workers:{NUM_WORKERS}")

X_indices = np.arange(len(_df_train)); y_labels = _df_train['label_encoded'].values
skf = StratifiedKFold(n_splits=_n_splits, shuffle=True, random_state=42)
fold_results = []; best_val_auc = -1.0; best_fold_idx = 0

with mlflow.start_run(run_name="NeuroDT-GPU-5Fold") as run:
    mlflow.log_params({'num_epochs':_n_epochs,'lr':LEARNING_RATE,'batch_size':BATCH_SIZE,
                       'n_splits':_n_splits,'use_amp':USE_AMP,'scheduler':'OneCycleLR',
                       'patience':EARLY_STOPPING_PATIENCE})

    for fold_idx,(train_idx,val_idx) in enumerate(skf.split(X_indices,y_labels)):
        print(f"\n{'='*55}\n  FOLD {fold_idx+1} / {_n_splits}\n{'='*55}")
        ckpt_path=BEST_MODEL_DIR/f'best_model_fold{fold_idx+1}.pth'

        # ── Resume check: is this fold already done? ─────────────────────
        resume_ckpt = None
        if ckpt_path.exists():
            _existing = torch.load(ckpt_path, map_location=device)
            if _existing.get('fold_complete'):
                print(f"  ✓ Fold {fold_idx+1} already complete "
                      f"(epoch {_existing['epoch']}, AUC={_existing['val_auc']:.4f}) — skipping.")
                fold_results.append({'fold':fold_idx+1,'best_val_loss':_existing['val_loss'],
                                      'best_val_auc':_existing['val_auc'],'ckpt_path':str(ckpt_path),
                                      'scaler':_existing['scaler']})
                if _existing['val_auc']>best_val_auc: best_val_auc=_existing['val_auc']; best_fold_idx=fold_idx
                continue
            else:
                resume_ckpt = _existing
                print(f"  ↻ Resuming Fold {fold_idx+1} from epoch {_existing['epoch']+1} "
                      f"(last val_loss={_existing['val_loss']:.4f}, AUC={_existing['val_auc']:.4f})")

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
        # OneCycleLR: 10% warmup → solves epoch-1 instability from CPU run
        scheduler=torch.optim.lr_scheduler.OneCycleLR(
            optimizer,max_lr=LEARNING_RATE,steps_per_epoch=len(train_loader),
            epochs=_n_epochs,pct_start=0.10,anneal_strategy='cos',
            div_factor=10.0,final_div_factor=1e4)
        criterion=nn.CrossEntropyLoss(weight=class_weights_tensor)
        scaler_amp=torch.cuda.amp.GradScaler(enabled=USE_AMP)
        best_val_loss=float('inf'); patience_count=0; start_epoch=1

        if resume_ckpt is not None:
            fold_model.load_state_dict(resume_ckpt['model_state_dict'])
            if 'optimizer_state_dict' in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in resume_ckpt:
                scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
            if 'scaler_amp_state_dict' in resume_ckpt:
                scaler_amp.load_state_dict(resume_ckpt['scaler_amp_state_dict'])
            best_val_loss=resume_ckpt['val_loss']
            start_epoch=resume_ckpt['epoch']+1
            if start_epoch>_n_epochs:
                print(f"  Fold {fold_idx+1} checkpoint is already past the last epoch — treating as complete.")
                resume_ckpt['fold_complete']=True
                torch.save(resume_ckpt, ckpt_path)
                fold_results.append({'fold':fold_idx+1,'best_val_loss':resume_ckpt['val_loss'],
                                      'best_val_auc':resume_ckpt['val_auc'],'ckpt_path':str(ckpt_path),
                                      'scaler':resume_ckpt['scaler']})
                if resume_ckpt['val_auc']>best_val_auc: best_val_auc=resume_ckpt['val_auc']; best_fold_idx=fold_idx
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
                    vl_probs+=torch.softmax(logits,1).cpu().tolist()
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
            mlflow.log_metrics({f'f{fold_idx+1}_vl_loss':vl_loss,f'f{fold_idx+1}_vl_auc':vl_auc,
                                 f'f{fold_idx+1}_tr_loss':tr_loss},step=fold_idx*_n_epochs+epoch)

            if vl_loss<best_val_loss:
                best_val_loss=vl_loss; patience_count=0
                torch.save({'fold':fold_idx+1,'epoch':epoch,'model_state_dict':fold_model.state_dict(),
                            'val_loss':vl_loss,'val_auc':vl_auc,'scaler':fold_scaler,'label_map':label_map,
                            'optimizer_state_dict':optimizer.state_dict(),
                            'scheduler_state_dict':scheduler.state_dict(),
                            'scaler_amp_state_dict':scaler_amp.state_dict(),
                            'fold_complete':False,
                            'training_config':{'scheduler':'OneCycleLR','batch_size':BATCH_SIZE,
                                               'use_amp':USE_AMP}},ckpt_path)
            else:
                patience_count+=1
                if patience_count>=EARLY_STOPPING_PATIENCE:
                    print(f'  Early stopping at epoch {epoch}'); break

        ckpt=torch.load(ckpt_path,map_location=device)
        ckpt['fold_complete']=True
        torch.save(ckpt, ckpt_path)
        fold_results.append({'fold':fold_idx+1,'best_val_loss':ckpt['val_loss'],
                              'best_val_auc':ckpt['val_auc'],'ckpt_path':str(ckpt_path),'scaler':fold_scaler})
        if ckpt['val_auc']>best_val_auc: best_val_auc=ckpt['val_auc']; best_fold_idx=fold_idx
        print(f"  Fold {fold_idx+1} best: vl_loss={ckpt['val_loss']:.4f}  AUC={ckpt['val_auc']:.4f}")
        gc.collect()
        if IS_GPU: torch.cuda.empty_cache()

aucs=[r['best_val_auc'] for r in fold_results]
print(f"\n{'='*55}")
print(f"  5-FOLD CV SUMMARY (GPU)")
print(f"  Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"  Best fold: {best_fold_idx+1}  (AUC={best_val_auc:.4f})")
for r in fold_results: print(f"  Fold {r['fold']}: AUC={r['best_val_auc']:.4f}")
print(f"{'='*55}")
mlflow.log_metrics({'cv_mean_auc':np.mean(aucs),'cv_std_auc':np.std(aucs)})
