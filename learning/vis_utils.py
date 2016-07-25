from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np

def plotSSCurve(sensitivity, specificity, people=[], title="", show_legend=False, textcolor='#4D5B66', plot_results=True):
    """SS curve for Publications.

    Args:
        sensitivity (numpy.array): length-N array of sensitivity points.
        specificity (numpy.array): length-N array of specificity points.
        people (list): List of lists. Each list entry is two float values:
            [persons_sensitivity, persons_specificity]

    Returns
        the handle to the figure, for further plotting.
    """
    area = auc(sensitivity, specificity)
    print 'The AUC is %s' % area

#   textcolor = 'black'
#   textcolor = 'darkgrey'
#   textcolor = '#4D5B66'
    textsize = 24
    rcParams = {
            'axes.grid' : False,
            'font.family' : 'sans-serif',
            'text.color' : textcolor,
            'axes.labelcolor' : textcolor,
            'axes.labelsize' : textsize,
            'axes.titlesize' : textsize,
            'axes.facecolor' : 'white',
            'axes.linewidth' : 3,
            'axes.prop_cycle' : plt.cycler('color', ['blue', 'black', '#5BC0DE', 'blue']),
            'figure.figsize' : (8,8),
            'xtick.color' : textcolor,
            'xtick.labelsize' : 20,
            'xtick.major.pad' : 15,
            'ytick.color' : textcolor,
            'ytick.labelsize' : 20,
            'ytick.major.pad' : 15,
            'legend.fontsize' : 20,
            }
    with plt.rc_context(rcParams):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.plot(sensitivity, specificity, label='Algorithm: AUC=%0.2f' % area, linewidth=4)
        xlabel = 'Sensitivity'
        ylabel = 'Specificity'
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.05])
        plt.yticks([0.0, 1.0], ["0", "1"])
        plt.xticks([0.0, 1.0], ["0", "1"])
        plt.axhline(y=1.0, xmin=0, xmax=0.95, color='k', zorder=-34, ls='dashed')
        plt.axhline(y=0.9, xmin=0, xmax=0.95, color='k', zorder=-34, ls='dashed')
        plt.axhline(y=0.8, xmin=0, xmax=0.95, color='k', zorder=-34, ls='dashed')
        plt.axvline(x=1.0, ymin=0, ymax=0.95, color='k', zorder=-34, ls='dashed')
        plt.axvline(x=0.9, ymin=0, ymax=0.95, color='k', zorder=-34, ls='dashed')
        plt.axvline(x=0.8, ymin=0, ymax=0.95, color='k', zorder=-34, ls='dashed')
        area = "%0.2f" % area
        title = title  + '\n'
        plt.title(title)
        for i, person in enumerate(people):
            if i == 0:
                plt.plot(person[0], person[1], 'o', color='red', zorder=-32, markersize=10, label='Dermatologists (%d)' % len(people))
            else:
                plt.plot(person[0], person[1], 'o', color='red', zorder=-32, markersize=10)
        if len(people) > 0:
            avg_sensitivity = np.mean(np.array(people)[:,0])
            avg_specificity = np.mean(np.array(people)[:,1])
            std_sensitivity = np.std(np.array(people)[:,0])
            std_specificity = np.std(np.array(people)[:,1])
            print 'Average sensitivity=%0.2f +- %0.2f' % (avg_sensitivity, std_sensitivity)
            print 'Average specificity=%0.2f +- %0.2f' % (avg_specificity, std_specificity)
            plt.plot(avg_sensitivity, avg_specificity, 'D', color='green', markersize=10, label='Average Dermatologist')
            plt.errorbar(avg_sensitivity, avg_specificity, xerr=std_sensitivity, yerr=std_specificity,
                    color='green', markersize=10, elinewidth=3)

        if show_legend:
            plt.legend(loc='lower left', numpoints= 1)
        if plot_results:
            plt.show()
    return fig, ax

