"""
Delete all model files script
WARNING: This will delete all .pt and .json files in the models directory!
"""

from pathlib import Path

models_dir = Path('models')

# Count and delete
deleted_count = 0

print("Deleting model files...")
print("-" * 80)

for file in sorted(models_dir.rglob('*.pt')):
    print(f"Deleting PT: {file.relative_to(models_dir.parent)}")
    file.unlink()
    deleted_count += 1

for file in sorted(models_dir.rglob('*.json')):
    print(f"Deleting JSON: {file.relative_to(models_dir.parent)}")
    file.unlink()
    deleted_count += 1

print("-" * 80)
print(f"Total files deleted: {deleted_count}")
print("Done!")
