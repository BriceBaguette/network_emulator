import matplotlib.pyplot as plt

# Define the x and y data
x_labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-10', '10-11', '11-12', '12-13', 
            '13-14', '14-15', '15-16', '16-17', '17-18', '18-19', '19-20', '20-21', '21-22', '22-23', 
            '23-24', '24-25', '25-26', '26-27', '27-28', '28-29', '29-30', '30-300']
y_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 487, 0, 0, 0, 113, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Create a bar plot
plt.figure(figsize=(12, 6))
plt.bar(x_labels, y_values, color='blue')
plt.ylim(0,600)
# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Add labels and title
plt.ylabel('Number of packet')
plt.xlabel('Bin template')
plt.title('Histogram of the number of packets in each bin')

# Show the plot
plt.tight_layout()
plt.show()
