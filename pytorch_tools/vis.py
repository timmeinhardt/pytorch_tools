"""Visdom Visualization."""
import numpy as np
import torch
from visdom import Visdom
import logging


logging.getLogger('visdom').setLevel(logging.CRITICAL)


class BaseVis(object):

    def __init__(self, viz_opts, update_mode='append', env=None, win=None, resume=False):
        self.viz_opts = viz_opts
        self.update_mode = update_mode
        self.win = win
        if env is None:
            env = 'main'
        self.viz = Visdom(env=env)
        # if resume first plot should not update with replace
        self.removed = not resume

    def close(self):
        if self.win is not None:
            self.viz.close(win=self.win)
            self.win = None


class LineVis(BaseVis):
    """Visdom Line Visualization Helper Class."""

    def plot(self, y_data, x_label):
        """Plot given data.

        Appends new data to exisiting line visualization.
        """
        update, opts = self.update_mode, None
        # update mode must be None the first time or after plot data was removed
        if self.removed:
            update, opts = None, self.viz_opts
            self.removed = False

        if isinstance(x_label, list):
            Y = torch.Tensor(y_data)
            X = torch.Tensor(x_label)
        else:
            y_data = [d.cpu() if torch.is_tensor(d)
                      else torch.tensor(d)
                      for d in y_data]

            Y = torch.Tensor(y_data).unsqueeze(dim=0)
            X = torch.Tensor([x_label])

        win = self.viz.line(
            X=X,
            Y=Y,
            opts=self.viz_opts,
            win=self.win,
            update=update)

        if self.win is None:
            self.win = win
        self.viz.save([self.viz.env])

    def reset(self):
        if self.win is not None:
            self.viz.line(X=None, Y=None, win=self.win, name='', update='remove')
            self.removed = True


class SurfVis(BaseVis):
    """Visdom Surface Visualization Helper Class."""

    def __init__(self, *args, **kwargs):
        super(SurfVis, self).__init__(*args, **kwargs)
        self.X_hist = []

    def plot(self, X):
        """Plot given data.

        Appends new data to exisiting surface visualization.
        """
        #if self.win is None:
        #    self.X_hist = []
        self.X_hist.append(X)
        if len(self.X_hist) < 2:
            return

        X_hist = torch.stack(self.X_hist).numpy().astype(np.float64)

        self.win = self.viz.contour(
            X=X_hist,
            opts=self.viz_opts,
            win=self.win, )
        self.viz.save([self.viz.env])

    def close(self):
        super(SurfVis, self).close()
        self.X_hist = []


class ImgVis(BaseVis):
    """Visdom Image Visualization Helper Class."""

    def plot(self, images):
        """Plot given images."""

        images = [img.data if isinstance(img, torch.autograd.Variable)
                  else img for img in images]
        images = [img.squeeze(dim=0) if len(img.size()) == 4
                  else img for img in images]

        self.win = self.viz.images(
            tensor=images,
            opts=self.viz_opts,
            win=self.win, )
        self.viz.save([self.viz.env])
