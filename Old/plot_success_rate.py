import matplotlib.pyplot as plt
import numpy as np

def plot_success_rate():
    """
    Create a visualization of Recipe Assistant performance metrics
    """
    # Simulated success rate metrics
    categories = [
        'Recipe Accuracy', 
        'Language Understanding', 
        'Cooking Instructions', 
        'Cultural Relevance', 
        'Overall Performance'
    ]
    
    # Simulated success rates (out of 100)
    success_rates = [92, 88, 95, 90, 91]
    
    # Create color gradient
    colors = plt.cm.Greens(np.linspace(0.5, 0.9, len(categories)))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, success_rates, color=colors)
    
    # Customize the plot
    plt.title('Indonesian Recipe Assistant Performance Metrics', fontsize=16)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}%',
                 ha='center', va='bottom', fontsize=10)
    
    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.savefig('recipe_assistant_success_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

# Call the function to generate the plot
plot_success_rate()

print("Success rate visualization has been saved as 'recipe_assistant_success_rate.png'")