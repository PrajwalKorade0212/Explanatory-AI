# Explain the model's predictions using SHAP values
explainer = shap.TreeExplainer(random_forest_model)
shap_values = explainer.shap_values(X_test)

# Visualize the summary plot
shap.summary_plot(shap_values, X_test)