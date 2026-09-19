import json

with open("export.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total tasks:", len(data))

labels = set()
for task in data:
    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            if result.get("type") == "brushlabels":
                labels.update(result.get("value", {}).get("brushlabels", []))

print("Labels:")
for label in sorted(labels):
    print("-", label)

print("\nFirst image data:")
if data:
    print(data[0].get("data"))
