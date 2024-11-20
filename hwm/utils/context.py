# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import time 
import sys
from typing import List, Dict, Optional 

__all__=["EpochBar", "ProgressBar"] 

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
            'loss': 1.0, 'acc': 0.5, 'val_loss': 1.0, 'val_acc': 0.5}
        

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
            [f"{k}: {v:.4f}" for k, v in self.metrics.items()]
        )
        print()
        print(f"Best Metrics: {best_metric_display}")

    def update(self, step, epoch, step_metrics=None,
               is_epoch_over= False):
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
        if step_metrics is None: 
            step_metrics = {}
            
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
                    if "loss" in metric or "pss" in metric:
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
        self._display_progress(step, epoch, is_epoch_over= is_epoch_over)

    def _display_progress(self, step, epoch, is_epoch_over=False):
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
        # Exclude 'val_' metrics unless it's the final batch
        if is_epoch_over:
            # Include all metrics, assuming 'val_' metrics are present
            metric_display = " - ".join([f"{k}: {v:.4f}" for k, v in self.metrics.items()])
        else:
            # Exclude 'val_' metrics
            metric_display = " - ".join(
                [f"{k}: {v:.4f}" for k, v in self.metrics.items() if not ( 
                    k.startswith('val_') or k.find('twa')>=0)]
                )
            
        # Print the progress bar and metrics to the console
        sys.stdout.write(
            f"\r{step}/{self.steps_per_epoch} "
            f"[{progress_bar}] - {metric_display}"
        )
        sys.stdout.flush()
        

class ProgressBar:
    """
    ProgressBar is a context manager for displaying a customizable progress bar 
    similar to Keras' training progress display. It is designed to handle 
    epoch-wise and batch-wise progress updates, providing real-time feedback 
    on metrics such as loss and accuracy.

    .. math::
        \text{Progress} = \frac{\text{current step}}{\text{total steps}}

    Attributes
    ----------
    total : int
        Total number of epochs to be processed.
    prefix : str, optional
        String to prefix the progress bar display (default is empty).
    suffix : str, optional
        String to suffix the progress bar display (default is empty).
    length : int, optional
        Character length of the progress bar (default is 30).
    decimals : int, optional
        Number of decimal places to display for the percentage (default is 1).
    metrics : List[str], optional
        List of metric names to display alongside the progress bar 
        (default is ['loss', 'accuracy', 'val_loss', 'val_accuracy']).
    steps : Optional[int], optional
        Number of steps per epoch. If not provided, defaults to `total`.

    Methods
    -------
    __enter__()
        Initializes the progress bar context.
    __exit__(exc_type, exc_value, traceback)
        Finalizes the progress bar upon exiting the context.
    update(iteration, epoch=None, **metrics)
        Updates the progress bar with the current iteration and metrics.
    reset()
        Resets the progress bar to its initial state.

    Examples
    --------
    >>> from hwm.utils.context import ProgressBar
    >>> total_epochs = 5
    >>> batch_size = 100
    >>> with ProgressBar(total=total_epochs, 
    ...                 prefix='Epoch', 
    ...                 suffix='Complete', 
    ...                 length=50) as pbar:
    ...     for epoch in range(1, total_epochs + 1):
    ...         print(f"Epoch {epoch}/{total_epochs}")
    ...         for batch in range(1, batch_size + 1):
    ...             # Simulate metric values
    ...             metrics = {'loss': 0.1 * batch, 
    ...                        'accuracy': 0.95 + 0.005 * batch, 
    ...                        'val_loss': 0.1 * batch, 
    ...                        'val_accuracy': 0.95 + 0.005 * batch}
    ...             pbar.update(iteration=batch, epoch=epoch, **metrics)
    ...             time.sleep(0.01)

    >>> total_epochs = 3
    >>> batch_size = 100
    >>> epoch_data = [
    ...    {
    ...        'loss': 0.1235, 
    ...        'accuracy': 0.9700, 
    ...        'val_loss': 0.0674, 
    ...        'val_accuracy': 0.9840
    ...    },
    ...    {
    ...        'loss': 0.0917, 
    ...        'accuracy': 0.9800, 
    ...        'val_loss': 0.0673, 
    ...        'val_accuracy': 0.9845
    ...    },
    ...    {
    ...        'loss': 0.0623, 
    ...        'accuracy': 0.9900, 
    ...        'val_loss': 0.0651, 
    ...        'val_accuracy': 0.9850
    ...    },
    ... ]

    >>> with ProgressBar(
    ...    total=total_epochs, 
    ...    prefix="Steps", 
    ...    suffix="Complete", 
    ...    length=30, 
    ...    metrics=['loss', 'accuracy', 'val_loss', 'val_accuracy']
    ... ) as pbar:
    ...    for epoch in range(1, total_epochs + 1):
    ...        print(f"Epoch {epoch}/{total_epochs}")
    ...        for batch in range(1, batch_size + 1):
    ...            # Rotate through example data for simulation
    ...            current_data = epoch_data[batch % len(epoch_data)]
    ...           pbar.update(
    ...                iteration=batch, 
    ...                epoch=None, 
    ...                **current_data
    ...            )
    ...            time.sleep(0.01)  # Simulate processing delay
    ...        print()
    
    Notes
    -----
    - The progress bar dynamically updates in place, providing real-time 
      feedback without cluttering the console.
    - Metrics are tracked and the best metrics are displayed upon completion 
      of the training process.

    See Also
    --------
    tqdm : A popular progress bar library for Python.
    rich.progress : A rich library for advanced progress bar visualizations.

    References
    ----------
    .. [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., 
           Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). 
           Attention is all you need. In Advances in neural information 
           processing systems (pp. 5998-6008).
    """

    _default_metrics: List[str] = ['loss', 'accuracy', 'val_loss', 'val_accuracy']

    def __init__(
        self,
        total: int,
        prefix: str = '',
        suffix: str = '',
        length: int = 30,
        steps: Optional[int] = None,
        decimals: int = 1,
        metrics: Optional[List[str]] = None
    ):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
        self.decimals = decimals
        self.metrics: List[str] = metrics 
        self.iteration: int = 0
        self.steps: int = steps if steps is not None else total
        self.start_time: Optional[float] = None

        # Initialize best metrics to track improvements
        self.best_metrics_: Dict[str, float] = {}
        if self.metrics is not None: 
            self.metrics: List[str] = metrics if metrics is not None else self._default_metrics
            
            for metric in self.metrics:
                if "loss" in metric or "pss" in metric:
                    self.best_metrics_[metric] = float('inf')  # For minimizing metrics
                else:
                    self.best_metrics_[metric] = 0.0  # For maximizing metrics

    def __enter__(self):
        """
        Enters the runtime context related to this object.

        Returns
        -------
        ProgressBar
            The ProgressBar instance itself.
        """
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exits the runtime context and performs final updates.

        Parameters
        ----------
        exc_type : type
            The exception type.
        exc_value : Exception
            The exception instance.
        traceback : TracebackType
            The traceback object.
        """
        # Final update to reach 100% completion
        if self.metrics is not None:
            best_metric_display = " - ".join(
                [f"{k}: {v:.4f}" for k, v in self.best_metrics_.items()]
            )
            print()
            print(f"Best Metrics: {best_metric_display}")
            
        print()

    def update(self, iteration: int, epoch: Optional[int] = None, **metrics):
        """
        Updates the progress bar with the current iteration and metrics.

        Parameters
        ----------
        iteration : int
            The current iteration or batch number within the epoch.
        epoch : Optional[int], optional
            The current epoch number (default is None).
        **metrics : dict
            Arbitrary keyword arguments representing metric names and their 
            current values (e.g., loss=0.1, accuracy=0.95).
        """
        self.iteration = iteration
        progress = self._get_progress(iteration)
        time_elapsed = time.time() - self.start_time if self.start_time else 0.0
        self._print_progress(progress, epoch, time_elapsed, **metrics)

    def reset(self):
        """
        Resets the progress bar to its initial state.

        This method is useful for resetting the progress bar at the start 
        of a new epoch or training phase.
        """
        self.iteration = 0
        self.start_time = time.time()
        self._print_progress(0.0)

    def _get_progress(self, iteration: int) -> float:
        """
        Calculates the current progress as a float between 0 and 1.

        Parameters
        ----------
        iteration : int
            The current iteration or batch number within the epoch.

        Returns
        -------
        float
            The progress ratio, constrained between 0 and 1.
        """
        progress = iteration / self.steps
        return min(progress, 1.0)

    def _format_metrics(self, **metrics) -> str:
        """
        Formats the metrics for display alongside the progress bar.

        Parameters
        ----------
        **metrics : dict
            Arbitrary keyword arguments representing metric names and their 
            current values.

        Returns
        -------
        str
            A formatted string of metrics.
        """
        if self.metrics is not None: 
            formatted = ' - '.join(
                f"{metric}: {metrics.get(metric, 0):.{self.decimals}f}" 
                for metric in self.metrics
            )
            return formatted
        else : 
            return ''

    def _update_best_metrics(self, metrics: Dict[str, float]):
        """
        Updates the best observed metrics based on current values.

        For metrics related to loss or PSS, the best metric is the minimum 
        observed value. For other performance metrics, the best metric is 
        the maximum observed value.

        Parameters
        ----------
        metrics : Dict[str, float]
            A dictionary of current metric values.
        """
        if self.metrics is not None: 
            for metric, value in metrics.items():
                if "loss" in metric or "pss" in metric:
                    # Track minimum values for loss and PSS metrics
                    if value < self.best_metrics_.get(metric, float('inf')):
                        self.best_metrics_[metric] = value
                else:
                    # Track maximum values for other performance metrics
                    if value > self.best_metrics_.get(metric, 0.0):
                        self.best_metrics_[metric] = value
                
    def _print_progress(
        self, 
        progress: float, 
        epoch: Optional[int] = None,
        time_elapsed: Optional[float] = None, 
        **metrics
    ):
        """
        Prints the progress bar to the console.

        Parameters
        ----------
        progress : float
            Current progress ratio between 0 and 1.
        epoch : Optional[int], optional
            Current epoch number (default is None).
        time_elapsed : Optional[float], optional
            Time elapsed since the start of the progress (default is None).
        **metrics : dict
            Arbitrary keyword arguments representing metric names and their 
            current values.
        """
        completed = int(progress * self.length)  # Number of '=' characters
        remaining = self.length - completed  # Number of '.' characters

        if progress < 1.0:
            # Progress bar with '>' indicating current progress
            bar = '=' * completed + '>' + '.' * (remaining - 1)
        else:
            # Fully completed progress bar
            bar = '=' * self.length

        percent = f"{100 * progress:.{self.decimals}f}%"

        # Display epoch information if provided
        epoch_info = f"Epoch {epoch}/{self.total} " if epoch is not None else ''

        # Display time elapsed if provided
        time_info = (
            f" - ETA: {time_elapsed:.2f}s" 
            if time_elapsed is not None else ""
        )

        # Format and update best metrics
        metrics_info = self._format_metrics(**metrics)
        self._update_best_metrics(metrics)

        # Construct the full progress bar display string
        
        display = (
            f'\r{epoch_info}{self.prefix} {self.iteration}/{self.steps} '
            f'[{bar}] {percent} {self.suffix} {metrics_info}{time_info}'
        )

        # Output the progress bar to the console
        sys.stdout.write(display)
        sys.stdout.flush()