import tensorflow as tf

model_dir = r"C:\Users\User\Piglet_Detector\piglet_project\exported-model\saved_model"
loaded = tf.saved_model.load(model_dir)
infer = loaded.signatures["serving_default"]

print("\n=== INPUT SIGNATURE ===")
print(infer.structured_input_signature)

print("\n=== OUTPUT SIGNATURE ===")
print(infer.structured_outputs)

print("\nDone.")
