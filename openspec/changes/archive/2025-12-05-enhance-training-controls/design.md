# Design: Enhanced Training Controls

## Architecture Overview

This change extends the existing training system with three independent but complementary capabilities:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Annotation Window                           │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│  │   Training   │  │   TensorBoard   │  │     Compare      │  │
│  │     Menu     │─▶│     Manager     │  │     Models       │  │
│  └──────────────┘  └─────────────────┘  └──────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
            ┌────────────────────────────────────┐
            │       Training Dialog (Enhanced)    │
            ├────────────────────────────────────┤
            │  • Basic Parameters                │
            │  • Advanced Options (collapsible)  │
            │    - YOLO-specific                 │
            │    - Classifier-specific           │
            │    - Defect-specific              │
            │  • Pause/Resume/Cancel buttons     │
            │  • Auto-TensorBoard checkbox       │
            └────────────────┬───────────────────┘
                             │
                             ▼
            ┌────────────────────────────────────┐
            │      Training Worker (QThread)      │
            ├────────────────────────────────────┤
            │  • Pause/Resume signal handling    │
            │  • Cancel signal handling          │
            │  • Checkpoint save/load            │
            │  • Progress updates                │
            └────────────────┬───────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
┌─────────────────────────┐  ┌──────────────────────────┐
│  Trainers (with         │  │  TensorBoard             │
│  TensorBoard logging)   │  │  SummaryWriter           │
├─────────────────────────┤  ├──────────────────────────┤
│  • YOLO Trainer         │─▶│  • Scalar metrics        │
│  • View Classifier      │  │  • Confusion matrices    │
│  • Defect Classifier    │  │  • Sample predictions    │
│                         │  │  • PASS class monitoring │
└─────────────┬───────────┘  └──────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│                  Database                                │
├─────────────────────────────────────────────────────────┤
│  • model_versions (existing)                            │
│  • training_history (existing)                          │
│  • checkpoints (new: temp storage for pause/resume)     │
└─────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│          Model Comparison                                │
├─────────────────────────────────────────────────────────┤
│  • ModelComparator (backend)                            │
│  • ModelComparisonDialog (UI)                           │
│  • Metrics delta calculation                            │
│  • Recommendation engine                                │
└─────────────────────────────────────────────────────────┘
```

## Component Design

### 1. Enhanced Training Dialog

**Key Design Decisions**:

#### 1.1 Progressive Disclosure UI Pattern
```python
# Basic parameters (always visible)
┌─────────────────────────────────┐
│ Model Type: [Detection ▼]      │
│ Epochs: [100]                   │
│ Batch Size: [16]                │
│ Learning Rate: [0.001]          │
│ Device: [Auto ▼]                │
└─────────────────────────────────┘

# Advanced Options (collapsible, default collapsed)
▼ Advanced Options
┌─────────────────────────────────┐
│ YOLO Detection Options:         │
│  Image Size: [1004 ▼]           │
│  Optimizer: [Adam ▼]            │
│  Patience: [20]                 │
│  Warmup Epochs: [3]             │
│                                 │
│ OR                              │
│                                 │
│ Classifier Options:             │
│  Backbone: [ResNet18 ▼]         │
│  Scheduler: [StepLR ▼]          │
│  Dropout: [0.3]                 │
│  ...                            │
└─────────────────────────────────┘
```

**Rationale**:
- 80% of users will use default settings
- Advanced users can expand for full control
- Reduces cognitive load for beginners

#### 1.2 Dynamic Parameter Visibility

```python
def _update_advanced_options(self, model_type):
    """Show only relevant parameters based on model type."""
    self.yolo_params_widget.setVisible(model_type == "detection")
    self.classifier_params_widget.setVisible(model_type in ["view", "defect"])
    self.defect_specific_widget.setVisible(model_type == "defect")
```

**Rationale**:
- Prevents confusion from irrelevant parameters
- Cleaner UI
- Follows principle of least surprise

### 2. Pause/Resume/Cancel Architecture

**State Machine**:

```
                    ┌─────────┐
                    │  IDLE   │
                    └────┬────┘
                         │ start_clicked
                         ▼
                    ┌─────────┐
             ┌─────▶│ RUNNING │◀──────┐
             │      └────┬────┘       │
             │           │            │
             │ resume    │ pause      │
             │           ▼            │
             │      ┌─────────┐      │
             └──────│ PAUSED  │──────┘
                    └────┬────┘
                         │ cancel
                         ▼
                    ┌─────────┐
                    │CANCELLED│
                    └─────────┘
```

**Implementation Strategy**:

```python
class TrainingWorker(QThread):
    def __init__(self):
        self.state = TrainingState.IDLE
        self.checkpoint_manager = CheckpointManager()

    def run(self):
        self.state = TrainingState.RUNNING

        for epoch in range(start_epoch, total_epochs):
            # Check for pause signal
            if self.state == TrainingState.PAUSED:
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    scheduler=self.scheduler
                )
                self._wait_for_resume()

            # Check for cancel signal
            if self.state == TrainingState.CANCELLED:
                self._cleanup_and_exit()
                return

            # Train one epoch
            self._train_epoch(epoch)

    def pause(self):
        """Thread-safe pause signal."""
        self.state = TrainingState.PAUSED

    def resume(self):
        """Thread-safe resume signal."""
        self.state = TrainingState.RUNNING
        self.resume_event.set()

    def cancel(self):
        """Thread-safe cancel signal."""
        self.state = TrainingState.CANCELLED
```

**Checkpoint Format**:

```python
{
    'version': '1.0',
    'model_type': 'defect_classifier',
    'epoch': 45,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'scheduler_state_dict': {...},
    'train_loss_history': [...],
    'val_loss_history': [...],
    'best_metric': 0.85,
    'config': {...},  # Full training config for reproducibility
    'timestamp': '2025-11-07T14:30:00'
}
```

**Atomic Save Strategy**:
```python
def save_checkpoint(self, checkpoint_data, path):
    """Atomic checkpoint save to prevent corruption."""
    temp_path = path + '.tmp'

    # Write to temp file
    torch.save(checkpoint_data, temp_path)

    # Verify integrity
    try:
        torch.load(temp_path)
    except Exception as e:
        os.remove(temp_path)
        raise CheckpointError(f"Checkpoint verification failed: {e}")

    # Atomic rename (OS-level atomic operation)
    os.replace(temp_path, path)
```

### 3. TensorBoard Integration Design

**Logging Architecture**:

```python
class TensorBoardLogger:
    """Centralized TensorBoard logging with smart batching."""

    def __init__(self, log_dir, model_type):
        self.writer = SummaryWriter(log_dir)
        self.model_type = model_type
        self.config = self._get_logging_config(model_type)

    def _get_logging_config(self, model_type):
        return {
            'scalar_every_epoch': True,
            'confusion_matrix_every_n': 5,  # Every 5 epochs
            'sample_predictions_every_n': 10,
            'histograms_every_n': 10,
            'log_gradients': False  # Too expensive for production
        }

    def log_epoch(self, epoch, metrics):
        """Log all epoch metrics."""
        # Scalars (lightweight, every epoch)
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'{self.model_type}/{name}', value, epoch)

        # Images/matrices (expensive, periodic)
        if epoch % self.config['confusion_matrix_every_n'] == 0:
            if 'confusion_matrix' in metrics:
                fig = plot_confusion_matrix(metrics['confusion_matrix'])
                self.writer.add_figure('Confusion_Matrix', fig, epoch)
                plt.close(fig)  # Free memory

    def log_pass_class_monitoring(self, epoch, metrics):
        """Special logging for PASS class (minority class)."""
        pass_metrics = extract_pass_class_metrics(metrics)

        self.writer.add_scalar('PASS/Recall', pass_metrics['recall'], epoch)
        self.writer.add_scalar('PASS/Precision', pass_metrics['precision'], epoch)
        self.writer.add_scalar('PASS/F1', pass_metrics['f1'], epoch)
        self.writer.add_scalar('PASS/FalseNegatives', pass_metrics['fn_count'], epoch)
```

**Performance Optimization**:

1. **Lazy Image Encoding**:
   ```python
   # Don't create images unless we're logging them
   if epoch % config['sample_predictions_every_n'] == 0:
       sample_imgs = create_prediction_grid(predictions)
       self.writer.add_images('Predictions', sample_imgs, epoch)
   ```

2. **Async Writes** (optional):
   ```python
   from concurrent.futures import ThreadPoolExecutor

   executor = ThreadPoolExecutor(max_workers=1)

   def async_log_figure(self, tag, figure, epoch):
       """Log figure asynchronously to avoid blocking training."""
       executor.submit(self.writer.add_figure, tag, figure, epoch)
   ```

3. **Batch Histogram Logging**:
   ```python
   # Log all model parameters in one call
   if epoch % config['histograms_every_n'] == 0:
       for name, param in model.named_parameters():
           self.writer.add_histogram(f'Parameters/{name}', param.data, epoch)
   ```

### 4. Model Comparison Architecture

**Backend Design**:

```python
class ModelComparator:
    """Compare multiple model versions with intelligent analysis."""

    def compare(self, model_type, version_ids):
        # Load all model metadata
        models = [self._load_model_metadata(vid) for vid in version_ids]

        # Calculate baseline (first model or active model)
        baseline = next((m for m in models if m['is_active']), models[0])

        # Compute deltas
        comparison = {
            'baseline': baseline,
            'models': models,
            'deltas': self._compute_deltas(baseline, models),
            'ranking': self._rank_models(models),
            'recommendation': self._generate_recommendation(models)
        }

        return comparison

    def _compute_deltas(self, baseline, models):
        """Compute metric differences vs baseline."""
        deltas = []
        for model in models:
            if model == baseline:
                deltas.append({})  # No delta for baseline
                continue

            delta = {}
            for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'mAP50']:
                if metric_name in baseline['metrics'] and metric_name in model['metrics']:
                    delta[metric_name] = {
                        'absolute': model['metrics'][metric_name] - baseline['metrics'][metric_name],
                        'relative': ((model['metrics'][metric_name] / baseline['metrics'][metric_name]) - 1) * 100
                    }
            deltas.append(delta)

        return deltas

    def _generate_recommendation(self, models):
        """Generate human-readable recommendation."""
        # Sort by primary metric (mAP for detection, balanced_accuracy for defect)
        primary_metric = self._get_primary_metric(models[0]['model_type'])
        sorted_models = sorted(models, key=lambda m: m['metrics'].get(primary_metric, 0), reverse=True)

        best = sorted_models[0]
        recommendation = f"Recommended: {best['version_name']}\n\n"

        # Explain why
        reasons = []
        if best['metrics'][primary_metric] > sorted_models[1]['metrics'][primary_metric]:
            improvement = (best['metrics'][primary_metric] - sorted_models[1]['metrics'][primary_metric]) * 100
            reasons.append(f"• {improvement:.1f}% better {primary_metric}")

        if best.get('inference_time_ms', float('inf')) < sorted_models[1].get('inference_time_ms', float('inf')):
            speed_diff = sorted_models[1].get('inference_time_ms', 0) - best.get('inference_time_ms', 0)
            reasons.append(f"• {speed_diff:.0f}ms faster inference")

        recommendation += "\n".join(reasons)
        return recommendation
```

**UI Design - Comparison Table**:

```
┌──────────────────────────────────────────────────────────────────┐
│ Model Comparison - Defect Classifier                        [X] │
├──────────────────────────────────────────────────────────────────┤
│ Metric             │ v1 (ACTIVE)  │ v2           │ Δ v2 vs v1   │
│────────────────────┼──────────────┼──────────────┼──────────────│
│ Balanced Accuracy  │ 0.852        │ 0.834        │ -1.8% ▼     │
│ Overall Accuracy   │ 0.891        │ 0.885        │ -0.6% ▼     │
│────────────────────┼──────────────┼──────────────┼──────────────│
│ PASS Recall        │ 0.733        │ 0.800        │ +6.7% ▲     │
│ PASS Precision     │ 0.917        │ 0.857        │ -6.0% ▼     │
│ PASS F1            │ 0.815        │ 0.828        │ +1.3% ▲     │
│────────────────────┼──────────────┼──────────────┼──────────────│
│ 晃動 Recall        │ 0.896        │ 0.883        │ -1.3% ▼     │
│ 沖線 Recall        │ 0.875        │ 0.850        │ -2.5% ▼     │
│ 碰觸 Recall        │ 0.912        │ 0.897        │ -1.5% ▼     │
│────────────────────┼──────────────┼──────────────┼──────────────│
│ Model Size (MB)    │ 18.3         │ 18.5         │ +0.2        │
│ Training Time      │ 1h 45m       │ 1h 52m       │ +7m         │
│ Inference (ms)     │ 23           │ 25           │ +2ms ▼      │
├──────────────────────────────────────────────────────────────────┤
│ 📊 Recommendation:                                               │
│ ┌────────────────────────────────────────────────────────────┐  │
│ │ v2 shows improved PASS recall (+6.7%) at the cost of       │  │
│ │ slightly lower overall accuracy (-1.8%).                   │  │
│ │                                                             │  │
│ │ Trade-off: Better minority class detection vs overall acc. │  │
│ │                                                             │  │
│ │ Recommended: v2 if PASS detection is critical             │  │
│ │ Recommended: v1 (current) for balanced performance        │  │
│ └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│ [View Confusion Matrices] [View TensorBoard] [Set v2 Active]    │
└──────────────────────────────────────────────────────────────────┘
```

## Trade-offs and Alternatives

### Trade-off 1: UI Complexity vs Control

**Decision**: Use collapsible advanced options (default collapsed)
**Alternative considered**: Separate "Basic" and "Advanced" modes with mode switcher
**Rationale**: Progressive disclosure is more intuitive; users can explore gradually

### Trade-off 2: Checkpoint Storage Location

**Decision**: Store in `checkpoints/` directory, NOT in database
**Alternative considered**: Store checkpoint blobs in database
**Rationale**:
- Checkpoints can be 100s of MB (model weights)
- SQLite performance degrades with large blobs
- File system is more appropriate for large binary data
- Database still tracks checkpoint metadata

### Trade-off 3: TensorBoard Logging Frequency

**Decision**: Scalars every epoch, images/matrices every 5-10 epochs
**Alternative considered**: Log everything every epoch
**Rationale**:
- Balance between detail and performance
- Scalars are cheap (<1% overhead)
- Images/matrices are expensive (can add 10-20% overhead)
- Most users check TensorBoard after training, not real-time

### Trade-off 4: Model Comparison Limit

**Decision**: Maximum 4 models can be compared simultaneously
**Alternative considered**: Unlimited comparison
**Rationale**:
- Table becomes unreadable with >4 columns
- Cognitive load increases exponentially
- 4 models covers 99% of real-world use cases (current vs 1-3 alternatives)

## Implementation Sequence

**Phase 1**: Training Controls (can start immediately)
- Expand TrainingDialog parameters
- Add pause/resume/cancel to TrainingWorker
- Implement CheckpointManager

**Phase 2**: TensorBoard Integration (can run in parallel with Phase 1)
- Add SummaryWriter to trainers
- Implement scalar logging
- Implement visual logging

**Phase 3**: Model Comparison (depends on Phase 1 completing)
- Implement ModelComparator backend
- Create ComparisonDialog UI
- Integrate with annotation window

**Parallelization Opportunity**: Phase 1 and Phase 2 have no dependencies and can be developed simultaneously.

## Testing Strategy

### Unit Tests
- CheckpointManager: save/load/verify integrity
- ModelComparator: delta calculation, ranking logic
- TensorBoardLogger: metric formatting

### Integration Tests
- Pause training → verify checkpoint saved
- Resume training → verify state restored correctly
- Cancel training → verify cleanup complete
- Model comparison → verify metrics match database

### End-to-End Tests
- Full training run with pause/resume cycle
- TensorBoard logs appear correctly
- Compare 3 models and verify recommendation

### Performance Tests
- Training speed with/without TensorBoard (target: <5% overhead)
- Checkpoint save time (target: <2 seconds)
- Model comparison response time (target: <1 second for 4 models)
