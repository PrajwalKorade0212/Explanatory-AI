from lime import lime_tabular

# Create a LIME explainer
explainer = lime_tabular.LimeTabularExplainer(training_data=X_train, mode='regression')

# Choose an index 'i' for the instance you want to explain
i = 0  # You can change this index according to your dataset

# Explain a prediction
explanation = explainer.explain_instance(X_test[i], random_forest_model.predict)

# Visualize the explanation
explanation.show_in_notebook()