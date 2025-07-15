import numpy as np
from types import SimpleNamespace
from view.analytics import Analytics

 
def test_analytics():
    shape = (100, 100)
    raw_data = np.random.rand(*shape)
    ground_truth = (np.random.rand(*shape) > 0.5).astype(np.uint8)
    
    class DummyConfig:
        def __init__(self, derivator, threshold=0.5):
            self.methods = SimpleNamespace(derivator=derivator)
            self.segmentation = SimpleNamespace(threshold=threshold)
    class DummyExperiment:
        def __init__(self, derivator):
            self.enhanced = np.clip(raw_data + np.random.normal(0, 0.1, raw_data.shape), 0, 1)
            self.segmented = (self.enhanced > 0.5).astype(np.uint8)
            self.config = DummyConfig(derivator)

    # Création de plusieurs expériences factices
    experiments = [DummyExperiment(f'Method_{i}') for i in range(6)]

    # Instanciation d'Analytics
    analytics = Analytics(display_mode=True)  # Pas d'affichage pendant le test

    # Test display_metrics
    # metrics_output = analytics.display_metrics(experiments, ground_truth)

    # # Test display_histograms
    fig_hist = analytics.display_histograms(experiments, raw_data, ground_truth)

    # # Test display_curves
    # fig_curves = analytics.display_curves(experiments, ground_truth)

    # # Test display_views
    # plots, titles, types = analytics.display_views(experiments, ground_truth, raw_data)

