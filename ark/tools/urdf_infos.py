
import xml.etree.ElementTree as ET



# Copy the path to your robots urdf file here
path = "/nfs/rlteam/sarthakdas/ark/examples/robots/viper_300s/vx300s.urdf"


# Load your URDF file
tree = ET.parse(path)
root = tree.getroot()

# Initialize a list to hold formatted joint info
joint_details = []

# Iterate over all joints and collect relevant info
for i, joint in enumerate(root.findall('joint')):
    joint_info = {}
    joint_info['Index'] = i
    joint_info['Name'] = joint.get('name')
    joint_info['Type'] = joint.get('type')

    # Get the parent and child link names
    joint_info['Parent Link'] = joint.find('parent').get('link')
    joint_info['Child Link'] = joint.find('child').get('link')

    # If joint has limits (revolute or prismatic joints), get the limits
    if joint_info['Type'] in ['revolute', 'prismatic']:
        limit = joint.find('limit')
        if limit is not None:
            joint_info['Lower Limit'] = limit.get('lower', 'N/A')
            joint_info['Upper Limit'] = limit.get('upper', 'N/A')
            joint_info['Effort Limit'] = limit.get('effort', 'N/A')
            joint_info['Velocity Limit'] = limit.get('velocity', 'N/A')
        else:
            joint_info['Limits'] = 'No limits defined'

    joint_details.append(joint_info)

def format_number(value):
    try:
        return f"{float(value):.4f}"  # Limit to 4 decimal places
    except ValueError:
        return value  # Return the value as-is if it's not a number

# Print out the formatted joint details with adjusted spacing
print(f"{'Index':<6}{'Joint Name':<20}{'Type':<18}{'Parent Link':<25}{'Child Link':<25}{'Lower Limit':<15}{'Upper Limit':<15}{'Effort Limit':<15}{'Velocity Limit':<15}")
print("-" * 145)

for joint in joint_details:
    print(f"{joint['Index']:<6}{joint['Name']:<20}{joint['Type']:<18}{joint['Parent Link']:<30}{joint['Child Link']:<30}"
          f"{format_number(joint.get('Lower Limit', 'N/A')):<15}{format_number(joint.get('Upper Limit', 'N/A')):<15}"
          f"{format_number(joint.get('Effort Limit', 'N/A')):<15}{format_number(joint.get('Velocity Limit', 'N/A')):<15}")