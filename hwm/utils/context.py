# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import time 
import sys

__all__=["EpochBar"] 

class EpochBar:
    """
    A context manager class to display a training progress bar during model 
    training, similar to the Keras progress bar, showing real-time updates 
    on metrics and progress.

    This class is designed to provide an intuitive way to visualize training 
    progress, track metric improvements, and display training status across 
    epochs. The progress bar is updated dynamically at each training step 
    to reflect current progress within the epoch, and displays performance 
    metrics, such as loss and accuracy.

    Parameters
    ----------
    epochs : int
        Total number of epochs for model training. This determines the 
        number of iterations over the entire training dataset.
    steps_per_epoch : int
        The number of steps (batches) to process per epoch. It is the 
        number of iterations per epoch, corresponding to the number of 
        batches the model will process during each epoch.
    metrics : dict, optional
        Dictionary of metric names and initial values. This dictionary should 
        include keys as metric names (e.g., 'loss', 'accuracy') and 
        values as the initial values (e.g., `{'loss': 1.0, 'accuracy': 0.5}`). 
        These values are updated during each training step to reflect the 
        model's current performance.
    bar_length : int, optional, default=30
        The length of the progress bar (in characters) that will be displayed 
        in the console. The progress bar will be divided proportionally based 
        on the progress made at each step.
    delay : float, optional, default=0.01
        The time delay between steps, in seconds. This delay is used to 
        simulate processing time for each batch and control the speed at 
        which updates appear.

    Attributes
    ----------
    best_metrics_ : dict
        A dictionary that holds the best value for each metric observed 
        during training. This is used to track the best-performing metrics 
        (e.g., minimum loss, maximum accuracy) across the epochs.

    Methods
    -------
    __enter__ :
        Initializes the progress bar and begins displaying training 
        progress when used in a context manager.
    __exit__ :
        Finalizes the progress bar display and shows the best metrics 
        after the training is complete.
    update :
        Updates the metrics and the progress bar at each step of training.
    _display_progress :
        Internal method to display the training progress bar, including 
        metrics at the current training step.
    _update_best_metrics :
        Internal method that updates the best metrics based on the current 
        values of metrics during training.

    Formulation
    -----------
    The progress bar is updated at each step of training as the completion 
    fraction within the epoch:

    .. math::
        \text{progress} = \frac{\text{step}}{\text{steps\_per\_epoch}}

    The bar length is represented by:

    .. math::
        \text{completed} = \text{floor}( \text{progress} \times \text{bar\_length} )
    
    The metric values are updated dynamically and tracked for each metric. 
    For metrics that are minimized (like `loss`), the best value is updated 
    if the current value is smaller. For performance metrics like accuracy, 
    the best value is updated if the current value is larger.
    
    Example
    -------
    >>> from hwm.utils.context import EpochBar
    >>> metrics = {'loss': 1.0, 'accuracy': 0.5, 'val_loss': 1.0, 'val_accuracy': 0.5}
    >>> epochs, steps_per_epoch = 10, 20
    >>> with EpochBar(epochs, steps_per_epoch, metrics=metrics,
    >>>                          bar_length=40) as progress_bar:
    >>>     for epoch in range(epochs):
    >>>         for step in range(steps_per_epoch):
    >>>             progress_bar.update(step + 1, epoch + 1)

    Notes
    -----
    - The `update` method should be called at each training step to update 
      the metrics and refresh the progress bar.
    - The progress bar is calculated based on the completion fraction within 
      the current epoch using the formula:

    .. math::
        \text{progress} = \frac{\text{step}}{\text{steps\_per\_epoch}}

    - Best metrics are tracked for both performance and loss metrics, with 
      the best values being updated throughout the training process.

    See also
    --------
    - Keras Callbacks: Callbacks in Keras extend the training process.
    - ProgressBar: A generic progress bar implementation.
    
    References
    ----------
    .. [1] Chollet, F. (2015). Keras. https://keras.io
    """
    def __init__(self, epochs, steps_per_epoch, metrics=None, 
                 bar_length=30, delay=0.01):
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.bar_length = bar_length
        self.delay = delay
        self.metrics = metrics if metrics is not None else {
            'loss': 1.0, 'accuracy': 0.5, 'val_loss': 1.0, 'val_accuracy': 0.5}
        
        # Initialize best metrics to track improvements
        self.best_metrics_ = {}
        for metric in self.metrics:
            if "loss" in metric or "PSS" in metric:
                self.best_metrics_[metric] = float('inf')  # For minimizing metrics
            else:
                self.best_metrics_[metric] = 0.0  # For maximizing metrics


    def __enter__(self):
        """
        Initialize the progress bar and begin tracking training progress 
        when used in a context manager.

        This method sets up the display and prepares the progress bar to 
        begin showing the current epoch and step during the training process.
        """
        print(f"Starting training for {self.epochs} epochs.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Finalize the progress bar and display the best metrics at the end 
        of the training process.

        This method will be called after all epochs are completed and 
        will display the best observed metrics across the training process.
        """
        best_metric_display = " - ".join(
            [f"{k}: {v:.4f}" for k, v in self.best_metrics_.items()]
        )
        print("\nTraining complete!")
        print(f"Best Metrics: {best_metric_display}")

    def update(self, step, epoch, step_metrics={}):
        """
        Update the metrics and refresh the progress bar at each training 
        step.

        This method is responsible for updating the training progress, 
        calculating the current values for the metrics, and refreshing the 
        display.

        Parameters
        ----------
        step : int
            The current step (batch) in the training process.
        epoch : int
            The current epoch number.
        step_metrics : dict, optional
            A dictionary of metrics to update for the current step. If 
            provided, the values will override the default ones for that 
            step.
        """
        time.sleep(self.delay)  # Simulate processing time per step
    
        for metric in self.metrics:
            if step == 0:
                # Initialize step value for the first step
                step_value = self.metrics[metric]
            else:
                if step_metrics:
                    # Update step_value based on provided step_metrics
                    if metric not in step_metrics:
                        continue
                    default_value = (
                        self.metrics[metric] * step + step_metrics[metric]
                    ) / (step + 1)
                else:
                    # For loss or PSS metrics, decrease value over time
                    if "loss" in metric or "PSS" in metric:
                        # Decrease metric value by a small step
                        default_value = max(
                            self.metrics[metric], 
                            self.metrics[metric] - 0.001 * step
                        )
                    else:
                        # For performance metrics, increase value over time
                        # Here we can allow unlimited increase
                        self.metrics[metric] += 0.001 * step
                        default_value = self.metrics[metric]
    
            # Get the step value for the current metric
            step_value = step_metrics.get(metric, default_value)
            self.metrics[metric] = round(step_value, 4)  # Round to 4 decimal places
    
        # Update the best metrics and display progress
        self._update_best_metrics()
        self._display_progress(step, epoch)

    def _update_best_metrics(self):
        """
        Update the best metrics based on the current values observed for 
        each metric during training.

        This method ensures that the best values for each metric are tracked 
        by comparing the current value to the previously recorded best value. 
        For metrics like loss, the best value is minimized, while for 
        performance metrics, the best value is maximized.
        """
        for metric, value in self.metrics.items():
            if "loss" in metric or "PSS" in metric:
                # Track minimum values for loss and PSS metrics
                if value < self.best_metrics_[metric]:
                    self.best_metrics_[metric] = value
            else:
                # Track maximum values for other performance metrics
                if value > self.best_metrics_[metric]:
                    self.best_metrics_[metric] = value


    def _display_progress(self, step, epoch):
        """
        Display the progress bar for the current step within the epoch.
    
        This internal method constructs the progress bar string, updates 
        it dynamically, and prints the bar with the metrics to the console.
    
        Parameters
        ----------
        step : int
            The current step (batch) in the training process.
        epoch : int
            The current epoch number.
        """
        progress = step / self.steps_per_epoch  # Calculate progress
        completed = int(progress * self.bar_length)  # Number of '=' chars to display
        
        # The '>' symbol should be placed where the progress is at,
        # so it starts at the last position.
        remaining = self.bar_length - completed  # Number of '.' chars to display
        
        # If the progress is 100%, remove the '>' from the end
        if progress == 1.0:
            progress_bar = '=' * completed + '.' * remaining
        else:
            # Construct the progress bar string with the leading 
            # '=' and trailing dots, and the '>'
            progress_bar = '=' * completed + '>' + '.' * (remaining - 1)
        
        # Ensure the progress bar has the full length
        progress_bar = progress_bar.ljust(self.bar_length, '.')
        
        # Construct the display string for metrics
        metric_display = " - ".join([f"{k}: {v:.4f}" for k, v in self.metrics.items()])
        
        # Print the progress bar and metrics to the console
        sys.stdout.write(
            f"\r{step}/{self.steps_per_epoch} "
            f"[{progress_bar}] - {metric_display}"
        )
        sys.stdout.flush()
