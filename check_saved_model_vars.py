import tensorflow as tf

model_dir = "models/research/exported-model/saved_model"
print("Checking SavedModel:", model_dir)

loaded = tf.saved_model.load(model_dir)

print("\n=== SIGNATURES ===")
for key in loaded.signatures:
    print("Signature:", key)

infer = loaded.signatures["serving_default"]

print("\n=== INPUTS ===")
for name, tensor in infer.structured_input_signature[1].items():
    print(name, ":", tensor)

print("\n=== OUTPUTS ===")
for name, tensor in infer.structured_outputs.items():
    print(name, ":", tensor)

print("\nDone.")
