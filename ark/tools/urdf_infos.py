#!/usr/bin/env python3
import argparse
import xml.etree.ElementTree as ET


def format_number(value):
    try:
        return f"{float(value):.4f}"  # Limit to 4 decimal places
    except (ValueError, TypeError):
        return value  # Return the value as-is if it's not a number


def parse_urdf(path: str):
    # Load URDF file
    tree = ET.parse(path)
    root = tree.getroot()

    joint_details = []

    for i, joint in enumerate(root.findall('joint')):
        joint_info = {
            'Index': i,
            'Name': joint.get('name'),
            'Type': joint.get('type'),
            'Parent Link': joint.find('parent').get('link'),
            'Child Link': joint.find('child').get('link')
        }

        # Limits (for revolute/prismatic joints)
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

    return joint_details


def print_joint_table(joint_details):
    # Print header
    print(f"{'Index':<6}{'Joint Name':<20}{'Type':<18}{'Parent Link':<25}{'Child Link':<25}"
          f"{'Lower Limit':<15}{'Upper Limit':<15}{'Effort Limit':<15}{'Velocity Limit':<15}")
    print("-" * 145)

    # Print rows
    for joint in joint_details:
        print(f"{joint['Index']:<6}{joint['Name']:<20}{joint['Type']:<18}"
              f"{joint['Parent Link']:<30}{joint['Child Link']:<30}"
              f"{format_number(joint.get('Lower Limit', 'N/A')):<15}"
              f"{format_number(joint.get('Upper Limit', 'N/A')):<15}"
              f"{format_number(joint.get('Effort Limit', 'N/A')):<15}"
              f"{format_number(joint.get('Velocity Limit', 'N/A')):<15}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse and display joint info from a URDF file."
    )
    parser.add_argument(
        "urdf_path",
        type=str,
        help="Path to the URDF file"
    )

    args = parser.parse_args()

    joints = parse_urdf(args.urdf_path)
    print_joint_table(joints)


if __name__ == "__main__":
    main()
