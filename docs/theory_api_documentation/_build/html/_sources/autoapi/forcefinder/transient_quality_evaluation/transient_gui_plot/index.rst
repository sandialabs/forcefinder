forcefinder.transient_quality_evaluation.transient_gui_plot
===========================================================

.. py:module:: forcefinder.transient_quality_evaluation.transient_gui_plot

.. autoapi-nested-parse::

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



Classes
-------

.. autoapisummary::

   forcefinder.transient_quality_evaluation.transient_gui_plot.SpectrogramCanvas
   forcefinder.transient_quality_evaluation.transient_gui_plot.LevelErrorCanvas


Functions
---------

.. autoapisummary::

   forcefinder.transient_quality_evaluation.transient_gui_plot.system_dependent_dpi


Module Contents
---------------

.. py:function:: system_dependent_dpi()

   Sets the DPI of the plots, depending on the detected OS.


.. py:class:: SpectrogramCanvas(parent=None, width=6, height=5, dpi=100)

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   The canvas the figure renders into.

   .. attribute:: figure

      A high-level figure instance.

      :type: `~matplotlib.figure.Figure`


   .. py:method:: plot_data(time, frequency, amplitude, label, colormap, colormap_limits, colorbar_label)

      Generates a spectrogram with Matplotlib pcolormesh.

      :param time: The time vector for the spectrogram.
      :type time: ndarray
      :param frequency: The frequency vector for the spectrogram.
      :type frequency: ndarray
      :param response_coordinate: The response coordinate array for the spectrogram.
      :type response_coordinate: coordinate_array
      :param amplitude: The amplitude of the spectrogram (in dB), it is organized [frequency axis, time axis].
      :type amplitude: ndarray
      :param label: The response coordinate label for the data being plotted.
      :type label: str
      :param colormap: The desired colormap for the spectrogram, this should correspond to an
                       available colormap in Matplotlib. The default is 'inferno'.
      :type colormap: str, optional
      :param colormap_limits: The limits for the colormap in the spectrogram. Must be provided as a list
                              of length two, where the first entry is the lower limit and the second entry
                              is the upper limit.
      :type colormap_limits: list, optional
      :param colorbar_label: The label for the colorbar in the spectrogram.
      :type colorbar_label: str



.. py:class:: LevelErrorCanvas(parent=None, width=6, height=5, dpi=100)

   Bases: :py:obj:`matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`


   The canvas the figure renders into.

   .. attribute:: figure

      A high-level figure instance.

      :type: `~matplotlib.figure.Figure`


   .. py:method:: plot_data(target_amplitude, target_time, predicted_amplitude, predicted_time, level_amplitude, level_time, data_axis_label, level_axis_label, label, linewidth)

      Plots data with matplotlib subplots.

      :param target_amplitude: The amplitude of the target time trace.
      :type target_amplitude: ndarray
      :param target_time: The time axis of the target time trace.
      :type target_time: ndarray
      :param predicted_amplitude: The amplitude of the predicted time trace.
      :type predicted_amplitude: ndarray
      :param predicted_time: The time axis of the predicted time trace.
      :type predicted_time: ndarray
      :param level_amplitude: The amplitude of the predicted time trace.
      :type level_amplitude: ndarray
      :param level_time: The time axis of the level time trace.
      :type level_time: ndarray
      :param data_axis_label: The y-axis label for the target and predicted data.
      :type data_axis_label: str
      :param level_axis_label: The y-axis label for the level.
      :type level_axis_label: str
      :param linewidth: The linewidth for the time traces in the plots.
      :type linewidth: float



