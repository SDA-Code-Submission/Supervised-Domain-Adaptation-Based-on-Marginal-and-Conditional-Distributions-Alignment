from Utilities.Imports import *
from Utilities.Configuration_Utils import *
from Utilities.Data_Utils import *
from Utilities.Adaption_Utils import *
from Utilities.Model_Utils import *
from Utilities.Training_Utils import *
from Utilities.Evaluation_Utils import *
from Utilities.Logging_Utils import *

parser = argparse.ArgumentParser()
parser = GetParser(parser)


def RunCofiguration(hp):
    print('--------- Beginning experiment %s ---------' % hp.ExpName)

    ## ----- Load dataset-----
    print('Loading datasets')
    train_loader, test_loader, val_loader, n_classes = get_datasets(hp)
    train_iter, val_iter = [get_iterator(x) for x in [train_loader, val_loader]]

    ## ----- Load model + optimizer initializations-----
    device = torch.device("cuda:%g" % hp.GPU if (torch.cuda.is_available() and hp.GPU >= 0) else "cpu")
    net = get_model(hp, n_classes, device)
    net.to(device)
    optimizer = get_optimizer(net, hp)

    ## ----- Initialize logger-----
    logger = MyLogger(n_classes)
    if hp.LogToWandb:
        get_initialized_wandb_logger(hp)

    ## ----- Train model----- 
    criterion = get_objective(hp)
    progress_bar = tqdm(range(hp.NumberOfBatches),
                        desc="Training model",
                        bar_format="{desc:20}{percentage:2.0f}{r_bar}")
    for batch in progress_bar:
        net.train()
        src_img, src_label, tgt_img, tgt_label = next(train_iter)
        src_img, tgt_img = (x.to(device, dtype=torch.float) for x in [src_img, tgt_img])

        src_label, tgt_label = (x.to(device, dtype=torch.long) for x in [src_label, tgt_label])
        (src_pred, src_feature), (tgt_pred, tgt_feature) = [net(x) for x in [src_img, tgt_img]]

        optimizer.zero_grad()

        # -----Explicit losses-----
        loss_s = criterion(src_pred, src_label)
        loss_t = criterion(tgt_pred, tgt_label)

        # -----DA losses-----
        if hp.Method == 'SDA_IO':
            loss_cdca = get_cdca_term(src_feature, tgt_feature, src_label, tgt_label, n_classes, criterion,hp)
            loss_uda = get_uda_term(src_feature, tgt_feature, hp)
            loss = hp.Coeffs[0] * loss_s + hp.Coeffs[1] * loss_t + hp.Coeffs[2] * loss_uda + hp.Coeffs[3] * loss_cdca
        if hp.Method == 'dSNE':
            loss_dsne = dsne_loss(src_feature, src_label, tgt_feature, tgt_label)
            loss = hp.Coeffs[0] * loss_s + hp.Coeffs[1] * loss_t + hp.Coeffs[4] * loss_dsne
        if hp.Method == 'CCSA':
            loss_csca = ccsa_loss(src_feature, tgt_feature,
                                  (src_label == tgt_label).float())
            loss = hp.Coeffs[0] * loss_s + hp.Coeffs[4] * loss_csca
        if hp.Method == 'NEM':
            loss_nem = nem_loss(tgt_img, src_feature, tgt_feature, src_label, tgt_label)
            loss = hp.Coeffs[0] * loss_s + hp.Coeffs[1] * loss_t + hp.Coeffs[4] * loss_nem
        if hp.Method == 'S-Only':
            loss =  loss_s
        if hp.Method == 'T-Only':
            loss =  loss_t
        if hp.Method == 'S+T':
            loss = hp.Coeffs[0] * loss_s + hp.Coeffs[1] * loss_t

        loss.backward()
        optimizer.step()

        # ----- Train Monitoring -----
        if hp.MonitorTraining:
            if n_classes>0:
                train_acc_src = get_acc(src_pred, src_label)
                train_acc_tgt = get_acc(tgt_pred, tgt_label)
            else:
                train_acc_src,train_acc_tgt= -1,-1
        # ----- Val Monitoring -----
        if batch % hp.ValMonitoringFactor == 0:
            with torch.no_grad():
                net.eval()
                src_img, src_label, tgt_img, tgt_label = next(val_iter)
                src_img, tgt_img = (x.to(device, dtype=torch.float) for x in [src_img, tgt_img])
                src_label, tgt_label = (x.to(device, dtype=torch.long) for x in [src_label, tgt_label])
                src_pred, tgt_pred = [net(x) for x in [src_img, tgt_img]]

                val_loss_s = criterion(src_pred, src_label)
                val_loss_t = criterion(tgt_pred, tgt_label)

                if n_classes > 0:
                    val_acc_src = get_acc(src_pred, src_label)
                    val_acc_tgt= get_acc(tgt_pred, tgt_label)
                else:
                    val_acc_src,val_acc_tgt=-1,-1
        else:
            val_loss_s, val_loss_t, val_acc_src, val_acc_tgt = -1, -1, -1, -1

        # ----- Logging -----
        if hp.MonitorTraining:
            logger.update(hp, batch, loss_s, loss_t, val_loss_s, val_loss_t,
                          train_acc_src, train_acc_tgt, val_acc_src, val_acc_tgt)
            logger.update_progress_bar(hp, batch, progress_bar)
        if hp.LogToWandb:
            log_to_wandb(hp, batch, loss_s, loss_t, val_loss_s, val_loss_t,
                         train_acc_src, train_acc_tgt, val_acc_src, val_acc_tgt)

    ## ----- Evaluate on test and log results-----
    src_test_metrics, tgt_test_metrics = [perf_eval(hp,net, test_loader, BatchLim=100, Domain=domain,
                                             Text='Evaluating test datapoints from the %s domain' % domain) for domain
                                    in ['Src', 'Tgt']]
    src_train_metrics, tgt_train_metrics= [perf_eval(hp,net, train_loader, BatchLim=100, Domain=domain,
                                               Text='Evaluating train datapoints from the %s domain' % domain) for
                                      domain in ['Src', 'Tgt']]

    log_results(hp, src_train_metrics, tgt_train_metrics, src_test_metrics, tgt_test_metrics)


##
if __name__ == "__main__":
    args = parser.parse_args()
    hp = GetConfFromArgs(args)
    RunCofiguration(hp)
