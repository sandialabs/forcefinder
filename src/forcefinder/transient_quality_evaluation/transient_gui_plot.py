"""
Defines GUI plotting tools for some transient error metrics. 

Copyright 2025 National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
import platform
import matplotlib.pyplot as plt
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QListWidget, QWidget, QHBoxLayout, QVBoxLayout)
except ImportError:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QListWidget, QWidget, QHBoxLayout, QVBoxLayout)
from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg as FigureCanvas,
                                               NavigationToolbar2QT as NavigationToolbar)

def system_dependent_dpi():
    """
    Sets the DPI of the plots, depending on the detected OS.
    """
    if platform.system() == 'Windows':
        return 100
    elif platform.system() == 'Darwin': #mac
        return 50
    else:
        return 75

class SpectrogramCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        # Create a Matplotlib figure and axis
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        plt.close(self.fig)
        super().__init__(self.fig)
        self.setParent(parent)
    
    def plot_data(self, time, frequency, amplitude, label, colormap, colormap_limits, colorbar_label):
        """
        Generates a spectrogram with Matplotlib pcolormesh.

        Parameters
        ----------
        time : ndarray 
            The time vector for the spectrogram.
        frequency : ndarray 
            The frequency vector for the spectrogram. 
        response_coordinate : coordinate_array 
            The response coordinate array for the spectrogram.
        amplitude : ndarray 
            The amplitude of the spectrogram (in dB), it is organized [frequency axis, time axis].
        label : str
            The response coordinate label for the data being plotted.
        colormap : str, optional
            The desired colormap for the spectrogram, this should correspond to an 
            available colormap in Matplotlib. The default is 'inferno'.
        colormap_limits : list, optional
            The limits for the colormap in the spectrogram. Must be provided as a list 
            of length two, where the first entry is the lower limit and the second entry
            is the upper limit.
        colorbar_label : str
            The label for the colorbar in the spectrogram.
        """
        # Storing the limits so they can be reused if data has already been plotted
        current_xlim = self.ax.get_xlim() if self.ax.has_data() else None
        current_ylim = self.ax.get_ylim() if self.ax.has_data() else None

        # need to delete the colorbar so a new one isn't made when the selected response coordinate changes
        if hasattr(self, 'cbar'):
            self.cbar.remove()
        self.ax.clear()  

        if colormap is None:
            colormap = 'inferno'

        if colormap_limits is None:
            mesh = self.ax.pcolormesh(time, frequency, amplitude, shading='auto', cmap=colormap)
        else:
            mesh = self.ax.pcolormesh(time, frequency, amplitude, shading='auto', cmap=colormap, vmin=colormap_limits[0], vmax=colormap_limits[1])
        
        self.ax.set_title(f'Response: {label}', fontsize=9)
        self.ax.set_xlabel('Time (s)', fontsize=9)
        self.ax.set_ylabel('Frequency (Hz)', fontsize=9)
        
        
        if current_xlim is None:
            self.ax.set_xlim(left=time.min(), right=time.max())
        else:
            self.ax.set_xlim(current_xlim)
        if current_ylim is None:
            self.ax.set_ylim(bottom=frequency.min(), top=frequency.max())
        else:
            self.ax.set_ylim(current_ylim)
        
        self.cbar = self.fig.colorbar(mesh, ax=self.ax)
        self.cbar.ax.set_ylabel(colorbar_label, fontsize=9)
        self.ax.tick_params(axis='both', labelsize=9)
        self.fig.set_tight_layout(tight=0.1)
        self.draw()


class SpectrogramGUI(QMainWindow):
    def __init__(self, data, colormap=None, colormap_limits=None, colorbar_label='Spectrogram Level'):
        """
        Initializes the basic GUI for plotting spectrogram with Matplotlib pcolormesh.

        Parameters
        ----------
        data : dict
            The spectrogram data to plot, packaged as a dictionary. It should have the 
            following keys:
                - time (ndarray) - The time vector for the spectrogram.
                - frequency (ndarray) - The frequency vector for the spectrogram. 
                - response_coordinate (coordinate_array) - The response 
                coordinate array for the spectrogram.
                - amplitude (ndarray) - The amplitude of the spectrogram, it is organized 
                [response coordinate, frequency axis, time axis].
        colormap : str, optional
            The desired colormap for the spectrogram, this should correspond to an 
            available colormap in Matplotlib. The default colormap depends on the 
            spectrogram type.
        colormap_limits : list, optional
            The limits for the colormap in the spectrogram. Must be provided as a list 
            of length two, where the first entry is the lower limit and the second entry
            is the upper limit.
        colorbar_label : str, optional
            The label for the colorbar in the spectrogram. The default is "Spectrogram level". 
        """
        super().__init__()
        self.data = data
        self.colormap = colormap
        self.colormap_limits = colormap_limits
        self.colorbar_label = colorbar_label
        self.setWindowTitle("Spectrogram Viewer")
        self.init_ui()
    
    def __repr__(self):
        return repr('Basic GUI plot for spectrograms')

    def init_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create a horizontal layout to hold the list widget and the plot
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)
        
        # List widget for response coordinates
        self.list_widget = QListWidget()
        for label in self.data['response_coordinate']:
            self.list_widget.addItem(str(label))
        self.list_widget.currentItemChanged.connect(self.on_item_changed)
        content_layout.addWidget(self.list_widget)

        self.canvas = SpectrogramCanvas(self, width=6, height=5, dpi=system_dependent_dpi())
        content_layout.addWidget(self.canvas, stretch=1)
        
        # Add the Navigation Toolbar for interactive capabilities
        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)

        # Select the first coordinate by default (if available)
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)
            self.update_plot(self.list_widget.currentItem().text())
        self.show()
    
    def on_item_changed(self, current):
        if current:
            self.update_plot(current.text())
    
    def update_plot(self, label):
        labels = list(self.data['response_coordinate'])
        index = labels.index(label)
        amplitude = self.data['amplitude'][index, :, :]
        time = self.data['time']
        frequency = self.data['frequency']
        self.canvas.plot_data(time, frequency, amplitude, label, self.colormap, self.colormap_limits, self.colorbar_label)

class LevelErrorCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        # Create a Matplotlib figure and axis
        self.fig, self.ax = plt.subplots(2,1, figsize=(width, height), dpi=dpi, sharex=True)
        plt.close(self.fig)
        super().__init__(self.fig)
        self.setParent(parent)
    
    def plot_data(self, target_amplitude, target_time, predicted_amplitude, predicted_time, 
                  level_amplitude, level_time, data_axis_label, level_axis_label, label, linewidth):
        """
        Plots data with matplotlib subplots.

        Parameters
        ----------
        target_amplitude : ndarray
            The amplitude of the target time trace.
        target_time : ndarray 
            The time axis of the target time trace.
        predicted_amplitude : ndarray 
            The amplitude of the predicted time trace.
        predicted_time : ndarray 
            The time axis of the predicted time trace.
        level_amplitude : ndarray 
            The amplitude of the predicted time trace.
        level_time : ndarray 
            The time axis of the level time trace.
        data_axis_label : str 
            The y-axis label for the target and predicted data.
        level_axis_label : str
            The y-axis label for the level.
        linewidth : float
            The linewidth for the time traces in the plots.
        """
        # Storing the limits so they can be reused if data has already been plotted
        current_xlim_data = self.ax[0].get_xlim() if self.ax[0].has_data() else None #only need limits for one x-axis since they're shared
        
        self.ax[0].clear()
        self.ax[1].clear()  

        self.ax[0].plot(target_time, target_amplitude, label='Target', linewidth=linewidth)
        self.ax[0].plot(predicted_time, predicted_amplitude, label='Predicted', linewidth=linewidth, linestyle=':')
        self.ax[0].grid()
        self.ax[0].legend()
        if data_axis_label is None:
            self.ax[0].set_ylabel('Time Data')
        else:
            self.ax[0].set_ylabel(data_axis_label)
        self.ax[0].set_title(f'Response: {label}', fontsize=9)
        
        self.ax[1].plot(level_time, level_amplitude, linewidth=linewidth)
        self.ax[1].set_xlabel('Time (s)')
        self.ax[1].grid()
        if level_axis_label is None:
            self.ax[1].set_ylabel('Level Error')
        else:
            self.ax[1].set_ylabel(level_axis_label)
        self.ax[1].set_xlabel('Time (s)', fontsize=9)
        
        # Setting the axis limits for the data portion of the plot
        if current_xlim_data is None:
            self.ax[0].set_xlim(left=min(target_time.min(),predicted_time.min(),level_time.min()), right=max(target_time.max(),predicted_time.max(),level_time.max()))
        else:
            self.ax[0].set_xlim(current_xlim_data)
        #if current_ylim_data is not None:
        #    self.ax[0].set_ylim(current_ylim_data)

        # Setting the axis limits for the level portion of the plot
        #if current_ylim_level is not None:
        #    self.ax[1].set_ylim(current_ylim_level)

        self.ax[0].tick_params(axis='both', labelsize=9)
        self.ax[1].tick_params(axis='both', labelsize=9)
        self.fig.set_tight_layout(tight=0.1)
        self.draw()

class LevelErrorGUI(QMainWindow):
    def __init__(self, target_data, predicted_data, level_error, data_axis_label=None, level_axis_label=None, linewidth=float(3)):
        """
        Initializes the basic GUI for plotting a level error and corresponding time traces.

        Parameters
        ----------
        target_data : TimeHistoryArray
            The target time history data for the comparison.
        predicted_data : TimeHistoryArray
            The data time history data to compare against the target.
        level_error : TimeHistoryArray
            The time varying level (e.g., RMS) comparison between the predicted and target 
            time histories. 
        data_axis_label : str, optional
            A string that is passed as the y-axis label for the time history portion
            of the plot.
        level_axis_label : str, optional
            A string that is passed as the y-axis label for the level portion of the 
            plot.
        linewidth : float, optional
            The linewidth to use for the curves in the plots. The default is 3.
        """
        super().__init__()
        comparison_coordinate = np.intersect1d(np.intersect1d(np.unique(target_data.response_coordinate), np.unique(predicted_data.response_coordinate)), np.unique(level_error.response_coordinate))

        self.data = {'response_coordinate':comparison_coordinate,
                     'target_abscissa':np.unique(target_data.abscissa),
                     'target_ordinate':target_data[comparison_coordinate[...,np.newaxis]].ordinate,
                     'predicted_abscissa':np.unique(predicted_data.abscissa),
                     'predicted_ordinate':predicted_data[comparison_coordinate[...,np.newaxis]].ordinate,
                     'level_abscissa':np.unique(level_error.abscissa),
                     'level_ordinate':level_error[comparison_coordinate[...,np.newaxis]].ordinate,
                     'data_axis_label':data_axis_label,
                     'level_axis_label':level_axis_label}
        self.linewidth = linewidth
        self.setWindowTitle("Time History and Level Error Viewer")
        self.init_ui()
    
    def __repr__(self):
        return repr('Basic GUI plot for transient level errors')

    def init_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create a horizontal layout to hold the list widget and the plot
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)
        
        # List widget for response coordinates
        self.list_widget = QListWidget()
        for label in self.data['response_coordinate']:
            self.list_widget.addItem(str(label))
        self.list_widget.currentItemChanged.connect(self.on_item_changed)
        content_layout.addWidget(self.list_widget)
        
        self.canvas = LevelErrorCanvas(self, width=6, height=5, dpi=system_dependent_dpi())
        content_layout.addWidget(self.canvas, stretch=1)
        
        # Add the Navigation Toolbar for interactive capabilities
        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)

        # Select the first coordinate by default (if available)
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)
            self.update_plot(self.list_widget.currentItem().text())
        self.show()
    
    def on_item_changed(self, current):
        if current:
            self.update_plot(current.text())
    
    def update_plot(self, label):
        labels = list(self.data['response_coordinate'])
        index = labels.index(label)
        target_amplitude = self.data['target_ordinate'][index, :]
        target_time = self.data['target_abscissa']
        predicted_amplitude = self.data['predicted_ordinate'][index, :]
        predicted_time = self.data['predicted_abscissa']
        level_amplitude = self.data['level_ordinate'][index, :]
        level_time = self.data['level_abscissa']
        self.canvas.plot_data(target_amplitude, target_time, predicted_amplitude, predicted_time, 
                              level_amplitude, level_time, self.data['data_axis_label'], self.data['level_axis_label'], label,
                              self.linewidth)