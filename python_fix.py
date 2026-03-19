
import re

NET_FILE = r"C:\Users\Asus ROG\Desktop\sumo-IA\sumo_rl\nets\construction\sumo-xml\net.net.xml"

print("Reading network file...")
with open(NET_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"Original size: {len(content)} bytes")

# Count occurrences
cable_car_count = content.count('cable_car')
subway_count = content.count('subway')
print(f"Found {cable_car_count} occurrences of 'cable_car'")
print(f"Found {subway_count} occurrences of 'subway'")

# Remove invalid classes
content = content.replace(' cable_car', '')
content = content.replace(' subway', '')

# Clean up any double spaces
content = re.sub(r'  +', ' ', content)

print(f"New size: {len(content)} bytes")

# Save fixed file
with open(NET_FILE, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Network file fixed!")

# Verify
with open(NET_FILE, 'r', encoding='utf-8') as f:
    test = f.read()
    
if 'cable_car' in test or 'subway' in test:
    print("❌ WARNING: Still found invalid classes!")
else:
    print("✅ VERIFIED: All invalid classes removed!")