import subprocess

# Run pip freeze command to get installed packages
result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE)

# Get the output and decode it from bytes to string
output = result.stdout.decode('utf-8')

# Write the requirements to a file
with open('requirements.txt', 'w') as f:
    f.write(output)

print("Requirements.txt file has been generated successfully.")
