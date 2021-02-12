from fastai.learner import Recorder, Callback
import torch

__all__ = ['CatchNaNCallback', 'GradientClipping']


class GradientClipping(Callback):
    "A `TrackerCallback` that reduces learning rate when a metric has stopped improving."
    def __init__(self, clip=0.0, factor=2., min_lr=1e-8):
        super().__init__()
        self.factor,self.min_lr,self.clip = factor,min_lr,clip

    def before_backward(self):
        if self.clip: torch.nn.utils.clip_grad_norm_(self.learn.model.parameters(), self.clip)

    def after_batch(self):
        "Compare the value monitored to its best score and reduce LR by `factor` if no improvement."
        pass
        # if torch.isinf(self.loss) or torch.isnan(self.loss):
        #     old_lr = self.opt.hypers[-1]['lr']
        #     for h in self.opt.hypers: h['lr'] = max(h['lr'] / self.factor, self.min_lr)
        #     if self.opt.hypers[-1]["lr"] < old_lr:
        #         print(f'Epoch {self.epoch}: reducing lr to {self.opt.hypers[-1]["lr"]}')
        #     self.learn.loss.clamp_(0, 0)


class CatchNaNCallback(Callback):
    run_before = Recorder

    """A `Callback` that terminates training if stored values are NaN."""
    def _check_nan_inf(self):
        """Test if `last_loss` is NaN and interrupts training."""
        # TODO: make the function check more values than only loss
        if torch.isinf(self.loss) or torch.isnan(self.loss):
            print(f'loss reached unsupported value: {self.loss}')
            # potential problem with grad_fn?
            # self.learn.loss = torch.zeros_like(self.loss, requires_grad=True)
            # TODO: make this callback decrease learning_rate
            self.learn.loss.clamp_(0, 0)
            self.learn.lr *= 0.5

    def before_batch(self): self._check_nan_inf()
    def after_pred(self): self._check_nan_inf()
    def after_loss(self): self._check_nan_inf()
    def before_backward(self): self._check_nan_inf()
    def after_backward(self): self._check_nan_inf()
    def after_step(self): self._check_nan_inf()
    def after_cancel_batch(self): self._check_nan_inf()
    def after_batch(self): self._check_nan_inf()
