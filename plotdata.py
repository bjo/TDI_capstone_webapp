
# The json theme was taken from Bokeh tutorial: https://github.com/bokeh/bokeh/blob/branch-3.0/examples/embed/embed_themed.py

from jinja2 import Template
import numpy as np

from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
# from bokeh.sampledata.iris import flowers
from bokeh.themes import Theme
from bokeh.transform import factor_cmap, factor_mark
# from bokeh.util.browser import view
from bokeh.models import ColumnDataSource

def return_plot_html(results: dict) -> dict:

	source = ColumnDataSource(data = dict(date = results['dates'], close = results['prices']))
	p = figure(height=300, width=800, x_axis_type="datetime", background_fill_color="#efefef")
	p.line('date', 'close', source = source)
	p.yaxis.axis_label = 'Price ($)'
	p.xaxis.axis_label = 'Date'

	# p = figure(height=300, width=800, tools="xpan", toolbar_location=None,
 #           x_axis_type="datetime", x_axis_location="above",
 #           background_fill_color="#efefef", x_range=(dates[1500], dates[2500]))

	theme = Theme(json={
	    'attrs': {
	        'Figure': {
	            'background_fill_color': '#3f3f3f',
	            'border_fill_color': '#3f3f3f',
	            'outline_line_color': '#444444'
	            },
	        'Axis': {
	            'axis_line_color': "white",
	            'axis_label_text_color': "white",
	            'major_label_text_color': "white",
	            'major_tick_line_color': "white",
	            'minor_tick_line_color': "white"
	            },
	        'Legend': {
	            'background_fill_color': '#3f3f3f',
	            'label_text_color': "white",
	        },
	        'Grid': {
	            'grid_line_dash': [6, 4],
	            'grid_line_alpha': .3
	            },
	        'Title': {
	            'text_color': "white"
	            }
	        }
	    })
	script, div = components(p, theme=theme)
	return_comps = {}
	return_comps['script'] = script
	return_comps['div'] = div
	return_comps['resources'] = INLINE.render()
	return return_comps