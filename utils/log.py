'''
Logger : Function for logging training losses and metrics.
         Save logs as the tensorboard log files in the training folder.
'''

class Logger :
    
    def __init__(self, ts_board) :
        self.ts_board = ts_board

    def step(self, model, iteration) :

        self.log_discriminator_loss(model, iteration)
        self.log_generator_loss(model, iteration)
        self.log_learning_rate(model, iteration)

    def log_fid_score(self, baseline_mag, proposed_mag, baseline_img, proposed_img, epoch) :

        self.ts_board.add_scalars('FID Score', {"Baseline Magnitude" : baseline_mag,
                                                "Ours Magnitude" : proposed_mag,
                                                "Baseline Image" : baseline_img,
                                                "Ours Image" : proposed_img,
                                               }, epoch)

    def log_discriminator_loss(self, model, iteration) :

        self.ts_board.add_scalars('Discriminator', {"Total Loss" : model.loss_D_total,
                                                    "Disc Loss" : model.loss_D,
                                                    "PSD Loss" : model.loss_SD,
                                                    }, iteration)

    def log_generator_loss(self, model, iteration) :
        
        self.ts_board.add_scalars('Adversarial Loss', {'Gen Loss' : model.loss_G_GAN,
                                                       'Spectral Loss' : model.loss_SG_GAN
                                                      }, iteration)
        self.ts_board.add_scalars('Generator', {"NCE Loss" : model.loss_NCE,
                                                "NCE Identity Loss" : model.loss_idt_NCE,
                                                "Spatial Identity" : model.loss_identity,
                                                "Spatial Fake" : model.loss_fake_identity,
                                                "Low Frequency Loss" : model.loss_LF,
                                                }, iteration)

    def log_classification_report(self, best_test_acc, epoch) :

        self.ts_board.add_scalar('Classification Accuracy', best_test_acc, epoch)

    def log_learning_rate(self, model, iteration) :
        
        self.ts_board.add_scalar('Learning Scheduler', model.optimizerG.param_groups[0]['lr'], iteration)
