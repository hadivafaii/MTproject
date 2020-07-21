import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns


def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()


def convert_time(time_in_secs):

    d = time_in_secs // 86400
    h = (time_in_secs - d * 86400) // 3600
    m = (time_in_secs - d * 86400 - h * 3600) // 60
    s = time_in_secs - d * 86400 - h * 3600 - m * 60

    print("\nd / hh:mm:ss   --->   %d / %d:%d:%d\n" % (d, h, m, s))


def plot_vel_field(data, fig_size=(14, 3), scale=None, save_file=None, estimate_center=False):
    if data.shape[-1] == 2:
        grd = data.shape[0]
        uu, vv = data[..., 0], data[..., 1]
    elif data.shape[0] == 2:
        grd = data.shape[-1]
        uu, vv = data[0, ...], data[1, ...]
    else:
        raise NotImplementedError
    xx, yy = np.mgrid[0:grd, 0:grd]
    cc = np.sqrt(np.square(uu) + np.square(vv))

    plt.figure(figsize=fig_size)
    plt.subplot(131)
    plt.imshow(uu)
    plt.colorbar()
    plt.xlim(-1, grd)
    plt.ylim(-1, grd)

    plt.subplot(132)
    plt.imshow(vv)
    plt.colorbar()
    plt.xlim(-1, grd)
    plt.ylim(-1, grd)

    plt.subplot(133)
    plt.quiver(xx, yy, uu, vv, cc, alpha=1, cmap='PuBu', scale=scale)
    plt.colorbar()
    plt.scatter(xx, yy, s=0.005)
    plt.xlim(-1, grd)
    plt.ylim(-1, grd)
    plt.axis('image')

    if save_file is not None:
        plt.savefig(save_file, facecolor='white')

    if estimate_center:
        ctr_y, ctr_x = np.unravel_index(np.argmax(cc), (grd, grd))
        plt.plot(ctr_y, ctr_x, 'r.', markersize=10)
        plt.show()

        return ctr_y, ctr_x

    plt.show()

    return


def make_gif(data_to_plot, frames=None, interval=120, dt=25, fig_sz=(8, 6), dpi=100, sns_style=None,
             cmap='Greys', mode='stim', scales=None, file_name=None, save_dir='./gifs/', row_n=None, col_n=None):
    if sns_style is not None:
        sns.set_style(sns_style)

    if frames is None:
        if mode in ['stim', 'stim_multi', 'velocity_field']:
            frames = np.arange(data_to_plot.shape[0])
        elif mode == 'subs':
            frames = np.arange(data_to_plot.shape[-2])

    if not os.path.isdir(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    if file_name is None:
        file_name = mode + '.gif'

    if mode == 'stim':
        # start the fig
        fig, _ax = plt.subplots()
        fig.set_size_inches(fig_sz)
        # fig.set_tight_layout(True)

        # find absmax for vmin/vmax and plot the first frame
        _abs_max = np.max(abs(data_to_plot))
        _plt = _ax.imshow(data_to_plot[0, ...],
                          cmap=cmap, vmin=-_abs_max, vmax=_abs_max)
        fig.colorbar(_plt)

        # start anim object
        anim = FuncAnimation(fig, _gif_update,
                             fargs=[_plt, _ax, data_to_plot, dt, None, None, None, mode],
                             frames=frames, interval=interval)
        anim.save(save_dir + file_name, dpi=dpi, writer='imagemagick')

    elif mode == 'stim_multi':
        if row_n is None:
            raise ValueError('For multiple stims must enter row_n')

        num_stim = data_to_plot.shape[-1]
        if col_n is None:
            col_n = int(np.ceil(num_stim / row_n))

        _plt_dict = {}

        fig, axes = plt.subplots(row_n, col_n)
        if fig_sz is None:
            fig.set_size_inches((col_n * 2, row_n * 2))
        else:
            fig.set_size_inches((fig_sz[0], fig_sz[1]))

        _abs_max = np.max(abs(data_to_plot))
        for ii in range(row_n):
            for jj in range(col_n):
                which_stim = ii * col_n + jj
                if which_stim >= num_stim:
                    continue

                axes[ii, jj].xaxis.set_ticks([])
                axes[ii, jj].yaxis.set_ticks([])
                tmp_plt = axes[ii, jj].imshow(data_to_plot[0, ..., which_stim], cmap=cmap)
                _plt_dict.update({'ax_%s_%s' % (ii, jj): tmp_plt})

        # start anim object
        anim = FuncAnimation(fig, _gif_update,
                             fargs=[fig, _plt_dict, data_to_plot, dt, None, row_n, col_n, mode],
                             frames=frames, interval=interval)
        anim.save(save_dir + file_name, dpi=dpi, writer='imagemagick')

    elif mode == 'velocity_field':
        k = data_to_plot.copy()
        k /= k.std()

        num_arr = k.shape[-1]
        grd = k.shape[1]

        if col_n is None:
            col_n = int(np.ceil(num_arr / row_n))

        _plt_dict = {}

        fig, axes = plt.subplots(row_n, col_n)
        if fig_sz is None:
            fig.set_size_inches((col_n * 2, row_n * 2))
        else:
            fig.set_size_inches((fig_sz[0], fig_sz[1]))

        xx, yy = np.mgrid[0:grd, 0:grd]

        for ii in range(row_n):
            for jj in range(col_n):
                which_arr = ii * col_n + jj
                if which_arr >= num_arr:
                    continue

                # estimate scale
                tmp = -1
                max_lag = -1
                for lag in range(k.shape[0]):
                    tmp_cc = np.sqrt(np.square(k[lag, ..., 0, which_arr])
                                     + np.square(k[lag, ..., 1, which_arr]))
                    if np.max(tmp_cc) > tmp:
                        tmp = np.max(tmp_cc)
                        max_lag = lag
                cc = np.sqrt(np.square(k[max_lag, ..., 0, which_arr])
                             + np.square(k[max_lag, ..., 1, which_arr]))

                # axes indexing
                if row_n > 1 and col_n > 1:
                    ax = axes[ii, jj]
                elif row_n > 1 and col_n == 1:
                    ax = axes[ii]
                elif row_n == 1 and col_n > 1:
                    ax = axes[jj]
                elif row_n == 1 and col_n == 1:
                    ax = axes
                else:
                    raise ValueError('row_n, col_n should be integers greater than or equal to 1')

                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])

                uu, vv = k[-1, ..., 0, which_arr], k[-1, ..., 1, which_arr]
                if scales is None:
                    scale = 8 * np.max(cc)
                else:
                    scale = scales[ii, jj]
                    cc = np.sqrt(np.square(uu) + np.square(vv))

                tmp_plt = ax.quiver(xx, yy, uu, vv, cc, alpha=1, scale=scale, cmap='PuBu')
                ax.scatter(xx, yy, s=0.01)

                ax.set_xlim(-1, grd)
                ax.set_ylim(-1, grd)

                ax.set_aspect('equal')

                ax.set_title('# %d' % which_arr)
                _plt_dict.update({'ax_%s_%s' % (ii, jj): tmp_plt})

        # start anim object
        anim = FuncAnimation(fig, _gif_update,
                             fargs=[fig, _plt_dict, k, dt, scales, row_n, col_n, mode],
                             frames=frames, interval=interval)
        anim.save(save_dir + file_name, dpi=dpi, writer='imagemagick')

    elif mode == 'subs':
        width_y, width_x, nlags, ker_n = data_to_plot.shape
        col_n = int(np.ceil(np.sqrt(ker_n)))
        row_n = int(np.ceil(ker_n / col_n))

        _plt_dict = {}

        fig, axes = plt.subplots(row_n, col_n)
        if fig_sz is None:
            fig.set_size_inches((col_n * 2, row_n * 2))
        else:
            fig.set_size_inches((fig_sz[0], fig_sz[1]))

        _abs_max = np.max(abs(data_to_plot))
        for ii in range(row_n):
            for jj in range(col_n):
                which_sub = ii * col_n + jj
                if which_sub >= ker_n:
                    continue

                axes[ii, jj].xaxis.set_ticks([])
                axes[ii, jj].yaxis.set_ticks([])
                tmp_plt = axes[ii, jj].imshow(data_to_plot[..., 0, which_sub], cmap=cmap)
                _plt_dict.update({'ax_%s_%s' % (ii, jj): tmp_plt})

        # start anim object
        anim = FuncAnimation(fig, _gif_update, fargs=[fig, _plt_dict, data_to_plot, dt, None, None, mode],
                             frames=frames, interval=interval)
        anim.save(save_dir + file_name, dpi=dpi, writer='imagemagick')

    else:
        raise ValueError('Not implemented yet.')

    plt.close()
    print('...your GIF is done! "%s" was saved at %s.' % (file_name, save_dir))


def _gif_update(tt, _fig_or_plt_like, _ax_like, data_to_plot, dt, scales=None, row_n=None, col_n=None, mode='stim'):
    nlags = data_to_plot.shape[0]
    time_remaining = np.rint((nlags - tt) * dt)
    lbl = '- {0} ms'.format(time_remaining)

    if mode == 'stim':
        _plt, _ax = _fig_or_plt_like, _ax_like

        _plt.set_data(data_to_plot[tt, :])
        _ax.set_xlabel(lbl)

        return _plt, _ax

    elif mode == 'stim_multi':
        if row_n is None:
            raise ValueError('For multiple stims must enter row_n')
        num_stim = data_to_plot.shape[-1]
        if col_n is None:
            col_n = int(np.ceil(num_stim / row_n))

        _fig, _plt_dict = _fig_or_plt_like, _ax_like

        for ii in range(row_n):
            for jj in range(col_n):
                which_stim = ii * col_n + jj
                if which_stim >= num_stim:
                    continue
                kk = data_to_plot[tt, ..., which_stim]
                _plt_dict['ax_%s_%s' % (ii, jj)].set_data(kk)
        _fig.suptitle(lbl, fontsize=50)

        return _plt_dict, _fig

    elif mode == 'velocity_field':
        if row_n is None:
            raise ValueError('For velocity fields must enter row_n')
        num_arr = data_to_plot.shape[-1]
        if col_n is None:
            col_n = int(np.ceil(num_arr / row_n))

        _fig, _plt_dict = _fig_or_plt_like, _ax_like

        for ii in range(row_n):
            for jj in range(col_n):
                which_arr = ii * col_n + jj
                if which_arr >= num_arr:
                    continue

                current_uu = data_to_plot[tt, ..., 0, which_arr]
                current_vv = data_to_plot[tt, ..., 1, which_arr]
                if scales is not None:
                    current_cc = np.sqrt(np.square(current_uu) + np.square(current_vv))
                    _plt_dict['ax_%s_%s' % (ii, jj)].set_UVC(current_uu, current_vv, current_cc)
                else:
                    _plt_dict['ax_%s_%s' % (ii, jj)].set_UVC(current_uu, current_vv)

        _fig.suptitle(lbl, fontsize=30)

        return _plt_dict, _fig

    elif mode == 'subs':
        ker_n = data_to_plot.shape[3]

        col_n = int(np.ceil(np.sqrt(ker_n)))
        row_n = int(np.ceil(ker_n / col_n))

        _fig, _plt_dict = _fig_or_plt_like, _ax_like

        for ii in range(row_n):
            for jj in range(col_n):
                which_sub = ii * col_n + jj
                if which_sub >= ker_n:
                    continue
                k = data_to_plot[..., tt, which_sub]
                _plt_dict['ax_%s_%s' % (ii, jj)].set_data(k)
        _fig.suptitle(lbl, fontsize=50)

        return _plt_dict, _fig
    else:
        raise ValueError('Not implemented yet.')
