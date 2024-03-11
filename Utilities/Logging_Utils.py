import wandb
import numpy as np

## Console logger
class MyLogger:
    def __init__(self,n_classes=10):
        self.n_classes=n_classes

        # loss
        self.train_losses_src= []
        self.train_losses_tgt= []
        self.val_losses_src= []
        self.val_losses_tgt= []

        # acc
        if not(n_classes==-1):
            self.train_accs_src= []
            self.train_accs_tgt= []
            self.val_accs_src= []
            self.val_accs_tgt = []

    def update(self,hp,batch,loss_s,loss_t,val_loss_s,val_loss_t,train_acc_src,train_acc_tgt,val_acc_src,val_acc_tgt):
        self.train_losses_src.append(loss_s.item())
        self.train_losses_tgt.append(loss_t.item())
        if batch % hp.ValMonitoringFactor == 0:
            self.val_losses_src.append(val_loss_s.item())
            self.val_losses_tgt.append(val_loss_t.item())
        if self.n_classes>0:
            self.train_accs_src.append(train_acc_src)
            self.train_accs_tgt.append(train_acc_tgt)
            if batch % hp.ValMonitoringFactor == 0:
                self.val_accs_src.append(val_acc_src)
                self.val_accs_tgt.append(val_acc_tgt)

    def update_progress_bar(self,hp,batch,progress_bar):
        if len(self.val_losses_src) > 10:
            if self.n_classes>0:
                progress_bar.set_postfix({'train_loss_src': np.mean(np.log10(self.train_losses_src[-10:])),
                                      'train_loss_tgt': np.mean(np.log10(self.train_losses_tgt[-10:])),
                                      'train_acc_src': np.mean(self.train_accs_src[-10:]),
                                      'train_acc_tgt': np.mean(self.train_accs_tgt[-10:]),
                                      'val_acc_tgt': self.val_accs_tgt[-1] if batch > hp.ValMonitoringFactor else -1})
            else:
                progress_bar.set_postfix({'train_loss_src': np.mean(np.log10(self.train_losses_src[-10:])),
                                          'train_loss_tgt': np.mean(np.log10(self.train_losses_tgt[-10:])),
                                          'val_loss_src': np.log10(self.val_losses_src[-1])if batch > hp.ValMonitoringFactor else -1,
                                          'val_loss_tgt': np.log10(self.val_losses_tgt[-1]) if batch > hp.ValMonitoringFactor else -1})

 ## W&B
def get_initialized_wandb_logger(hp):
    if hp.WadbUsername == 'EnterYourUserName':
        raise Exception("If \'LogToWandb\' is set to True, then it is required to "
                        "modify the username accordingly in line 15 in \'Utilities\Configuration_Utils.py\'")
    wandb.login()
    wandb.init(project=hp.ProjectName, entity=hp.WadbUsername)
    wandb.run.name = hp.ExpName
    config = wandb.config
    config.args = vars(hp)
    return config

def log_to_wandb(hp,batch,loss_s,loss_t,val_loss_s,val_loss_t,train_acc_src,train_acc_tgt,val_acc_src,val_acc_tgt):
    wandb.log({
        'SrcTrainLoss': loss_s.item(),
        'TgtTrainLoss': loss_t.item()})
    if hp.TaskObjective=='CE':
        wandb.log({
            'SrcTrainAcc': train_acc_src,
            'TgtTrainAcc': train_acc_tgt})
    if batch % hp.ValMonitoringFactor == 0:
        wandb.log({
            'TgtValLoss': val_loss_t,
            'SrcValLoss': val_loss_s})
        if hp.TaskObjective=='CE':
            wandb.log({
                'TgtValAcc': val_acc_tgt,
                'SrcValAcc': val_acc_src})
    return 1

## Log final results
def log_results(hp, src_train_metrics, tgt_train_metrics, src_test_metrics, tgt_test_metrics):
    if hp.TaskObjective=='CE':
        metric_name="Acc"
    else:
        metric_name = "MSE"

    print('Results for experiment: %s' % hp.ExpName)
    print('     Source domain:')
    print('         Train %s = %g' % (metric_name,np.mean(src_train_metrics)))
    print('         Test %s = %g' % (metric_name,np.mean(src_test_metrics)))
    print('     Target domain:')
    print('         Train %s=%g' % (metric_name,np.mean(tgt_train_metrics)))
    print('         Test %s=%g' % (metric_name,np.mean(tgt_test_metrics)))

    if hp.LogToWandb:
        wandb.run.summary["Source Train %s"%metric_name] = np.mean(src_train_metrics)
        wandb.run.summary["Target Train %s"%metric_name] = np.mean(tgt_train_metrics)
        wandb.run.summary["Source Test %s"%metric_name] = np.mean(src_test_metrics)
        wandb.run.summary["Target Test %s"%metric_name] = np.mean(tgt_test_metrics)