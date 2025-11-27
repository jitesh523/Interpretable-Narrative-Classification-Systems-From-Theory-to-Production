try:
    from lime.lime_text import LimeTextExplainer
except ImportError:
    LimeTextExplainer = None

class Explainer:
    def __init__(self, class_names=None):
        if LimeTextExplainer is None:
            raise ImportError("LIME is not installed. Please install 'lime' package.")
        self.explainer = LimeTextExplainer(class_names=class_names)
        
    def explain_instance(self, text, predict_proba_fn, num_features=10):
        exp = self.explainer.explain_instance(
            text, 
            predict_proba_fn, 
            num_features=num_features
        )
        return exp.as_list()
