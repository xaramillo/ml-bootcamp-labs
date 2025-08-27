import shap
import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.inspection import PartialDependenceDisplay

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def explain_shap(model, X_train, X_test, feature_names, model_name, outdir="int_exp_artifacts"):
    ensure_dir(outdir)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # Summary plot (bar)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    fname_bar = os.path.join(outdir, f"shap_summary_bar_{model_name}.png")
    plt.savefig(fname_bar, bbox_inches='tight')
    plt.close()
    # Summary plot (dot)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    fname_dot = os.path.join(outdir, f"shap_summary_dot_{model_name}.png")
    plt.savefig(fname_dot, bbox_inches='tight')
    plt.close()
    # Force plot (local)
    idx = 0
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value, shap_values[idx], X_test.iloc[idx], feature_names=feature_names, matplotlib=True, show=False)
    fname_force = os.path.join(outdir, f"shap_force_{model_name}_idx{idx}.png")
    plt.savefig(fname_force, bbox_inches='tight')
    plt.close()
    # Dependence plot (top feature)
    importances = np.abs(shap_values).mean(axis=0)
    top_feature_idx = np.argmax(importances)
    top_feature = feature_names[top_feature_idx]
    plt.figure()
    shap.dependence_plot(top_feature, shap_values, X_test, feature_names=feature_names, show=False)
    fname_dep = os.path.join(outdir, f"shap_dependence_{model_name}_{top_feature}.png")
    plt.savefig(fname_dep, bbox_inches='tight')
    plt.close()
    return [fname_bar, fname_dot, fname_force, fname_dep], idx, top_feature

def explain_lime(model, X_train, X_test, feature_names, class_names, idx, model_name, outdir="int_exp_artifacts"):
    ensure_dir(outdir)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=class_names,
        mode="classification"
    )
    exp = explainer.explain_instance(
        data_row=np.array(X_test.iloc[idx]),
        predict_fn=model.predict_proba
    )
    fname_lime = os.path.join(outdir, f"lime_explanation_{model_name}_idx{idx}.png")
    fig = exp.as_pyplot_figure()
    fig.savefig(fname_lime, bbox_inches='tight')
    plt.close(fig)
    return fname_lime

def explain_pdp(model, X_test, feature_names, top2_features, model_name, outdir="int_exp_artifacts"):
    ensure_dir(outdir)
    fig, ax = plt.subplots(figsize=(8, 6))
    display = PartialDependenceDisplay.from_estimator(
        model, X_test, features=top2_features, feature_names=feature_names, ax=ax
    )
    fname_pdp = os.path.join(outdir, f"pdp_{model_name}_{'_'.join([str(f) for f in top2_features])}.png")
    plt.savefig(fname_pdp, bbox_inches='tight')
    plt.close(fig)
    return fname_pdp