# Code for the "Inferring Keystrokes from Myo Armband Sensors" project
#
# Copyright (C) 2020-2021  Matthias Gazzari, Annemarie Mattmann
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
A collection of specific analysis plots for recreation.
"""

import itertools
import pathlib
import os
import re

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import norm

# function to convert matplotlib pts width to inches by
# https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

sns.set() # set back to default
FIGSIZE = set_size(452.295, fraction=0.5)
matplotlib.rcParams["figure.figsize"] = FIGSIZE
sns.set_context("paper", rc={"text.usetex": True, "font.family":"serif"})
tex_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    #"font.serif": ["Libertine"],
    "text.latex.preamble": r"\usepackage{libertine}\usepackage[libertine]{newtxmath}",
    "axes.labelsize": 9, # because of small caption size
    "axes.titlesize": 8,
    "font.size": 9, # because of small caption size
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}
plt.rcParams.update(tex_fonts)

#plt.rcParams.update({
#	"text.usetex": True,
#	"font.family": "serif",
#	"axes.labelsize": 10,
#	"font.size": 10,
#})

COLOR_PALETTE = "Set1"
GRAYSCALE_PALETTE = sns.color_palette("tab10", n_colors=10, desat=1)
BAR_HATCHES = ["","\\\\","//","xx","--",".","*","o"]
LINE_STYLES = ["solid", "dashed", "dotted", (0, (3, 1, 1, 1)), "dashdot", (0, (3, 5, 1, 5, 1, 5))]

def prepare_plot_paper(data, y, hue, col, row):
	# if hue is an array, create a column from the data
	if not isinstance(hue, str) and hasattr(hue, "__len__"):
		data = columns_to_hue(data, "hue", y, hue)
		hue = "hue"
	# sanitize input
	if col and len(data[col].unique()) <= 1:
		col = None
	if row and len(data[row].unique()) <= 1:
		row = None
	return data, hue, col, row

def columns_to_hue(data, hue, y, columns, keey_only_columns=None):
	"""
	Return a reshaped copy of the given data for use in the plot function to create hue plots.
	hue: A string that can be passed to the plot function as hue.
	y: A string that can be passed to the plot function as y (the y axis label).
	columns: An array of column labels. The values in these columns will be plotted on top of each
	other in one figure.
	keey_only_columns: An array of column labels. If this is given, the returned data will only
	include these columns and the new hue and y columns. If it is not given, the returned data will
	include all columns not within the given columns and the new hue and y columns.
	If it is an empty array, the returned data will only include the new hue and y columns.
	Example:
	>>> df
	   a  b  c  d
	0  1  3  5  7
	1  2  4  6  8
	>>> columns_to_hue(df, "bc", "value", ["b", "c"])
	   a  d bc  value
	0  1  7  b      3
	1  2  8  b      4
	2  1  7  c      5
	3  2  8  c      6
	>>> columns_to_hue(df, "bc", "value", ["b", "c"], ["a"])
	   a bc  value
	0  1  b      3
	1  2  b      4
	2  1  c      5
	3  2  c      6
	"""
	data = data.copy()
	if keey_only_columns is not None:
		data = data.melt(id_vars=keey_only_columns, value_vars=columns, var_name=hue, value_name=y)
	else:
		labels_not_in_columns = list(set(data) - set(columns))
		data = data.melt(id_vars=labels_not_in_columns, var_name=hue, value_name=y)
	return data

def plot_pdf(fig, title, path, filename):
	"""Maximize a given figure, store it or show its title."""
	#plt.tight_layout()
	# show maximized for qt backend
	if matplotlib.get_backend().startswith('Qt'):
		manager = plt.get_current_fig_manager()
		manager.window.showMaximized()
	# store
	if filename:
		pathlib.Path(path).mkdir(parents=True, exist_ok=True)
		plt.savefig(os.path.join(path, filename + ".pdf"), format="pdf", bbox_inches="tight")
	else:
		plt.suptitle(title, fontsize=22)

def sanitize_input(arg):
	try:
		return arg.replace("_", " ")
	except:
		return None

def plotf(func, data, x=None, y=None, hue=None, col=None, row=None, style=None,
		  title="", path="results/analysis/", filename="", png=False,
		  font_scale=1, **kwa_kind):
	"""Common plot function to use."""
	data, hue, col, row = prepare_plot_paper(data, y, hue, col, row)
	# sanitize input for latex
	try:
		data = data.replace({"_": " "}, regex=True)
		data.columns = data.columns.str.replace("_", " ", regex=True)
	except AttributeError:
		pass
	x = sanitize_input(x)
	y = sanitize_input(y)
	hue = sanitize_input(hue)
	col = sanitize_input(col)
	row = sanitize_input(row)
	style = sanitize_input(style)
	# store all values not common to all function as additional keyword arguments
	kwa_kind["y"] = y
	kwa_kind["hue"] = hue
	kwa_kind["col"] = col
	kwa_kind["row"] = row
	kwa_kind["style"] = style
	#fig = plt.figure(figsize=set_size(452.295, fraction=0.5)) # ACM large textwidth/2 to figsize
	fig = func(data, x=x, **kwa_kind)
	plot_pdf(fig, title, path, filename)


########################################### apply models ###########################################

def plot_truth_prediction(data, x, y, hue=None, row=None, style=None, palette=COLOR_PALETTE,
						  draw_line=None, spans=None, aspect=None, height=None,
						  **kwa_kind):
	"""Plot a seaborn relplot to compare truth and prediction data."""
	unique_rows = len(data[row].unique())
	fig, axes = plt.subplots(unique_rows, 1, figsize=FIGSIZE, sharex=True)#, sharey=True)
	fill_palette = itertools.cycle(sns.color_palette(palette))
	for i, ax in enumerate(axes):
		next_color = next(fill_palette)
		current_data = data[row].unique()[i]
		data_ax = data[data[hue] == current_data]
		# draw on separate axes to preserve figsize
		sns.lineplot(
			ax=ax,
			x=x, y=y, hue=hue, style=style, data=data_ax, palette=(next_color,),
			legend=False,
			dashes=False, ci=None
		)
		ax.set_title(current_data, y=0.6, fontsize=8)
		# adjust 0 and 1 positioning
		ax.set_ylim(-0.4, 1.4)
		# remove overlapping ylabels
		ax.set_ylabel("")
		# for each prediction: draw a window around the truth
		if spans and i > 0:
			for span in spans[i-1]:
				ax.axvspan(span[0], span[1], facecolor=sns.color_palette(palette)[i], alpha=0.2)
		# fill the area below each curve
		ax.fill_between(data_ax[x].values, data_ax[y].values, color=next_color, alpha=0.3)
		# draw a vertical line if requested
		if draw_line is not None:
			ax.axhline(y=draw_line, linestyle="--", linewidth=0.5, color="black", zorder=1)
	# add a centered ylabel, thanks to https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots/36542971
	#fig.add_subplot(111, frameon=False)
	#plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
	#plt.grid(False)
	#plt.ylabel(y)
	fig.text(0.03, 0.5, y, ha="center", va="center", rotation="vertical")

	return fig

def plot_metric_results_bars(data, x, y, hue=None, palette=COLOR_PALETTE, **kwa_kind):
	"""Plot a seaborn bar chart to show the metric results."""
	plt.figure()
	fig = sns.barplot(x=x, y=y, hue=hue, data=data, palette=palette, capsize=0, ci="sd")
	fig.set_ylim(0, 1)
	# remove legend header
	handles, labels = fig.get_legend_handles_labels()
	fig.legend(handles=handles[:], labels=labels[:])
	return fig

def plot_metric_results_bars_with_dodge(data, x, y, hue=None,  palette=COLOR_PALETTE, **kwa_kind):
	"""Plot a seaborn bar chart to show the metric results."""
	plt.figure()
	fig = sns.barplot(x=x, y=y, hue=hue, data=data, palette=palette, capsize=0, dodge=False, ci="sd")
	styles = data.drop_duplicates(subset=x).sort_values(by=x).reset_index()[hue]
	keys = data[hue].unique().tolist()
	for i, bar in enumerate(fig.patches):
		bar.set_hatch(BAR_HATCHES[keys.index(styles[i%len(styles)])])
	# fix legend entry hatch
	fig.patches[0].set_hatch(BAR_HATCHES[0])
	fig.set_ylim(0, 1)
	# remove legend header
	handles, labels = fig.get_legend_handles_labels()
	fig.legend(handles=handles[:], labels=labels[:])
	return fig

def plot_metric_results_cat(data, x, y, hue=None, style=None, col=None, palette=COLOR_PALETTE, **kwa_kind):
	"""Plot a seaborn grid of barplots to show the metric results."""
	fig = sns.catplot(
			x=x, y=y, hue=hue, col=col, data=data, palette=palette, #style=style,
			legend=False, legend_out=False,
			capsize=0.1, kind="bar"
		).set_titles("{col_name}").add_legend(title="")
	fig.set(ylim=(0, 1))
	return fig

def plot_speed_performance_scatter(data, x, y, hue=None, style=None, palette=COLOR_PALETTE, draw_line=None, **kwa_kind):
	"""Plot a seaborn scatterplot to correlate typing speed and model performance."""
	plt.figure()
	fig = sns.scatterplot(x=x, y=y, hue=hue, data=data, palette=palette, style=style)
	fig.set_ylim(-0.04, 1.04)
	if draw_line is not None:
		fig.axhline(y=draw_line, linestyle="--", linewidth=0.5, color="grey", zorder=1)
	# remove legend header
	handles, labels = fig.get_legend_handles_labels()
	fig.legend(handles=handles[0:], labels=labels[0:])
	return fig

def plot_speed_performance_lm(data, x, y, hue=None, style=None, palette=COLOR_PALETTE, aspect=None, height=None,
							  **kwa_kind):
	"""Plot a seaborn lmplot to correlate typing speed and model performance."""
	fig = sns.lmplot(
			x=x, y=y, hue=hue, data=data, palette=palette, style=style,
			legend=False, legend_out=False,
			aspect=aspect, height=height
		).add_legend(title="")
	fig.set(ylim=(-0.04, 1.04))
	return fig

def plot_temp_metric_line(data, x, y, hue=None, style=None, palette=COLOR_PALETTE, **kwa_kind):
	"""Plot a seaborn line chart to show the distance metric result."""
	plt.figure()
	fig = sns.lineplot(x=x, y=y, hue=hue, style=hue, data=data, palette=palette, ci="sd")
	fig.set_ylim(-0.04, 1.04)
	# remove legend header
	handles, labels = fig.get_legend_handles_labels()
	fig.legend(handles=handles[0:], labels=labels[0:])
	return fig

def plot_temp_metric_bin_mul_line(data, x, y, hue=None, style=None, palette=COLOR_PALETTE, **kwa_kind):
	"""Plot a seaborn line chart to show the distance metric result for both binary and multiclass
	predictions."""
	plt.figure()
	fig = sns.lineplot(x=x, y=y, hue=hue, style=hue, data=data, palette=palette, ci=None)
	fig.set_ylim(0, 1)
	# remove legend header
	handles, labels = fig.get_legend_handles_labels()
	fig.legend(handles=handles[0:], labels=labels[0:], loc="lower center")
	return fig

def plot_temp_metric_rel(data, x, y, hue=None, style=None, col=None, palette=COLOR_PALETTE,
						 **kwa_kind):
	"""Plot a grid of seaborn lineplots to show the distance metric results."""
	title = ("{col_name} " if col else "")
	fig = sns.relplot(
			x=x, y=y, col=col, hue=hue, data=data, palette=palette, style=style,
			legend=False,
			kind="line"
		).set_titles(title)
	fig.set(ylim=(-0.04, 1.04))
	# add legend
	if hue is not None:
		fig.add_legend(labels=data[hue].unique(), title="", frameon=True, loc=(0.05, 0))
	return fig

def plot_lags_dist(data, x, palette=COLOR_PALETTE, **kwa_kind):
	"""Plot a seaborn distribution plot to show a lag histogram."""
	plt.figure()
	fig = sns.histplot(x=x, data=data.reset_index(), palette=palette, kde=True)
	return fig

def plot_lags_box(data, x, y, hue=None, style=None, col=None, palette=COLOR_PALETTE, **kwa_kind):
	"""Plot a seaborn grid of boxplots to show the lags."""
	plt.figure()
	fig = sns.catplot(
			x=x, y=y, hue=hue, col=col, data=data, palette=palette, #style=style,
			legend=False, legend_out=False,
			flierprops=dict(marker="o"),
			kind="box"
		).set_titles("{col_name}").add_legend(title="")
	return fig

def stacked_bars(data, x, y, hue=None, stacks=None, palette=COLOR_PALETTE, **kwa_kind):
	"""Plot a stacked bar plot."""
	plt.figure()
	fill_palette = itertools.cycle(list(reversed(sns.color_palette(palette)))[-len(stacks)-2:-2])
	stacks = [el.replace("_", " ") for el in stacks]
	old_sort = ["".join(re.findall(r"\d+", stack)) for stack in stacks]
	stacks_sorted = {int(k):v for k,v in zip(old_sort,stacks)}
	stacks = dict(sorted(stacks_sorted.items(), key=lambda x: x[0], reverse=True))
	stacks = stacks.values()
	for i, stack in enumerate(stacks):
		next_color = next(fill_palette)
		y_stack = stack.replace("_", " ")
		fig = sns.barplot(x=x, y=y_stack, hue=hue, data=data, palette=(next_color,), ci=None, dodge=False)
	for i, bar in enumerate(fig.patches):
		bar.set_hatch(BAR_HATCHES[i//(data[x].nunique()*len(stacks)//2)])
	fig.set_ylim(0, 1)
	fig.set_ylabel(y)
	handles, labels = fig.get_legend_handles_labels()
	fig.legend(handles=handles[0::2], labels=stacks)
	return fig

def stacked_bars_hue(data, x, y, hue=None, stacks=None, palette=COLOR_PALETTE, **kwa_kind):
	"""Plot a stacked bar plot."""
	plt.figure()
	fill_palette = itertools.cycle(list(reversed(sns.color_palette(palette)))[-len(stacks)-2:-2])
	stacks = [el.replace("_", " ") for el in stacks]
	old_sort = ["".join(re.findall(r"\d+", stack)) for stack in stacks]
	stacks_sorted = {int(k):v for k,v in zip(old_sort,stacks)}
	stacks = dict(sorted(stacks_sorted.items(), key=lambda x: x[0], reverse=True))
	stacks = stacks.values()
	for stack in stacks:
		next_color = next(fill_palette)
		y_stack = stack.replace("_", " ")
		fig = sns.barplot(x=x, y=y_stack, hue=hue, data=data, palette=(next_color,), ci=None)
	fig.set_ylim(0, 1)
	fig.set_ylabel(y)
	handles, labels = fig.get_legend_handles_labels()
	fig.legend(handles=handles[0::4], labels=stacks)
	return fig


### exp_print_timing

def plot_pp_interval(data, x, hue=None, palette=COLOR_PALETTE, xlim=(-0.1,2.1), loc=None, drawline=None, **kwa_kind):
	legend_entries = list(data[hue].unique())
	plt.figure()
	fig = sns.ecdfplot(data=data, x=x, hue=hue, hue_order=data[hue].unique())
	# change linestyles
	for i, line in enumerate(fig.get_lines()):
		line.set_linestyle(LINE_STYLES[i])
	plt.ylabel("proportion")
	fig.set_xlim(xlim)
	fig.set_ylim((-0.04,1.04))
	if drawline is not None:
		fig.axvline(x=drawline, linestyle="--", linewidth=0.5, color="grey", zorder=1)
	# remove legend header and change legend loc and style
	handles, labels = fig.legend_.legendHandles, [t.get_text() for t in fig.legend_.get_texts()]
	for i, handle in enumerate(handles):
		line_styles = LINE_STYLES[:-(len(LINE_STYLES)-data[hue].nunique())]
		handle.set_linestyle(line_styles[::-1][i])
	fig.legend(handles=handles[:], labels=labels[:], loc=loc)
	return fig


### exp_sensor_data_words

def show_dtw_users(data, x, y, hue=None, style=None, palette=COLOR_PALETTE, xlim=(0,1.02), ylim=(0,1.02), **kwa_kind):
	sns.set_palette(["#d3d3d3"] + sns.color_palette(COLOR_PALETTE))
	plt.figure()
	fig = sns.scatterplot(x=x, y=y, hue=hue, data=data, style=style,
							alpha=0.7)
	#fig.set_xlim(xlim)
	#fig.set_ylim(ylim)
	#fig.set_xlabel(x,fontsize=10)
	# remove legend header and move legend to the right of the plot
	handles, labels = fig.get_legend_handles_labels()
	fig.legend(handles=handles[0:], labels=labels[0:])#, loc="center left", bbox_to_anchor=(1, 0.5))
	return fig


### analyse_sensors

def plot_sensor_difference(data, x, y, hue=None, palette=COLOR_PALETTE, **kwa_kind):
	plt.figure()
	fig = sns.barplot(x=x, y=y, hue=hue, data=data, palette=sns.color_palette(palette)[1:], capsize=0, ci="sd", dodge=True)
	for i, bar in enumerate(fig.patches):
		bar.set_hatch(BAR_HATCHES[i//data[x].nunique()])
	fig.set_ylim(0, 1)
	# remove legend header
	handles, labels = fig.get_legend_handles_labels()
	fig.legend(handles=handles[:], labels=labels[:])#, loc="lower right")
	return fig


## exp_time_lag_dist

def plot_time_lag(data, x, y=None, hue=None, palette=COLOR_PALETTE, **kwa_kind):
	plt.figure()
	fig = sns.distplot(
		data,
		label="peak time difference",
		axlabel = "lag [s]",
		fit=norm, kde=False,
		fit_kws={"color": "r", "label": "normal distribution fit"},
	)
	fig.set_ylabel("density")
	#plt.legend()
	return fig


## exp_clap_sync

def plot_clap_sync(data, x, y, hue, style=None, palette=COLOR_PALETTE, **kwa_kind):
	plt.figure()
	fig = sns.lineplot(data=data, x=x, y=y, hue=hue, style=style, palette=palette)
	# remove legend header
	handles, labels = fig.get_legend_handles_labels()
	fig.legend(handles=handles[0:], labels=labels[0:])
	return fig
