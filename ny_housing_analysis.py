# dont mind the link comments, i use org mode so they come from that.
# [[file:../code.org::*Data Trimming][Data Trimming:1]]
#!/usr/bin/env python3
import pandas as pd
import numpy as np

def trim_data():
    """Filter the raw dataset to remove outliers"""
    df = pd.read_csv('NY-House-Dataset.csv')
    price = df['PRICE']

    # Calculate bounds
    mean_price = price.mean()
    std_price = price.std(ddof=1)
    lower_bound = 150000
    upper_bound = mean_price + 3 * std_price

    # Filter data
    df_filtered = df[(price >= lower_bound) & (price <= upper_bound)]
    df_filtered = df_filtered[
    (df_filtered['BEDS'] > 0) & (df_filtered['BEDS'] <= 15)
&
    (df_filtered['BATH'] > 0) & (df_filtered['BATH'] <= 10)
]
    df_filtered.to_csv('NY-House-Dataset_Filtered.csv', index=False)

    print(f"Original data points: {len(df)}")
    print(f"Filtered data points: {len(df_filtered)}")
    print(f"Price range kept: ${lower_bound:,.2f} to ${upper_bound:,.2f}")
# Data Trimming:1 ends here

# [[file:../code.org::*Data Summary][Data Summary:1]]
def summarize_data():
    """Generate basic statistics about the dataset"""
    df = pd.read_csv('NY-House-Dataset_Filtered.csv')
    price = df['PRICE']

    print("\n=== Basic Statistics ===")
    print("Mean:\n", df.mean(numeric_only=True).round(3))
    print("\nMedian:\n", df.median(numeric_only=True).round(3))
    print("\nTotal Listings:", len(df))
    print(f"Price Range: ${price.min():,.2f} to ${price.max():,.2f}")
    print("\nCorrelation Matrix:\n", df.corr(numeric_only=True).round(2))
# Data Summary:1 ends here

# [[file:../code.org::*Plot Configuration][Plot Configuration:1]]
def configure_plots():
    import os
    """Set consistent plot styling"""
    import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True)
    plt.rcParams.update({
        "text.usetex": False,
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 300
    })
# Plot Configuration:1 ends here

# [[file:../code.org::*Price Distribution][Price Distribution:1]]
def plot_price_distribution():
    """Create histogram and log-normal fit of prices"""
    import matplotlib.pyplot as plt
    from scipy.stats import lognorm
    from matplotlib.ticker import FuncFormatter

    df = pd.read_csv('NY-House-Dataset_Filtered.csv')
    price_scaled = df['PRICE'] / 100_000  # Scale to 100k units

    # Fit log-normal distribution
    shape, loc, scale = lognorm.fit(price_scaled, floc=0)
    x = np.linspace(min(price_scaled), max(price_scaled), 1000)
    pdf = lognorm.pdf(x, shape, loc, scale)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(price_scaled, bins=100, density=True,
            color='lightgray', edgecolor='black', alpha=0.6,
            label='Histogram')
    ax.plot(x, pdf, 'r-', lw=2, label='Log-Normal Fit')
    ax.axvline(price_scaled.mean(), color='blue', linestyle='--',
               label=f'Mean: ${price_scaled.mean() * 100_000 / 1e6:.2f}M')

    # Formatting
    formatter = FuncFormatter(lambda val, _: f'${val * 100_000 / 1e6:.1f}M')
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xscale('log')
    ax.set_xlabel('Price (log scale, in millions)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Log-Normal Distribution of Housing Prices')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig('figures/lognormhist.png')
    plt.close()
# Price Distribution:1 ends here

# [[file:../code.org::*3D Location Plot][3D Location Plot:1]]
def plot_3d_location():
    """Create 3D scatter plot of prices by location"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import FuncFormatter

    df = pd.read_csv('NY-House-Dataset_Filtered.csv')
    price_scaled = df['PRICE'] / 100_000

    notable_locations = [
        ("Manhattan", 40.7831, -73.9712),
        ("Brooklyn", 40.6782, -73.9442),
        ("Queens", 40.7282, -73.7949),
        ("Bronx", 40.8448, -73.8648),
        ("Staten Island", 40.5795, -74.1502)
    ]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df['LATITUDE'], df['LONGITUDE'], price_scaled,
                   c=price_scaled, cmap='viridis', marker='o', s=3, alpha=0.8)

    # Formatting
    ax.zaxis.set_major_formatter(FuncFormatter(lambda val, _: f'${val * 100_000 / 1e6:.1f}M'))
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Price (in millions)', labelpad=15)
    ax.set_title('3D Scatter of Housing Prices by Location')

    # Label locations
    approx_price = np.percentile(price_scaled, 90)
    for name, lat, lon in notable_locations:
        ax.text(lat, lon, approx_price, name, color='grey',
               fontsize=14, fontweight='bold')

    ax.grid(True, linestyle='--', alpha=0.4)
    cb = fig.colorbar(sc, pad=0.1, shrink=0.6,
                     format=FuncFormatter(lambda val, _: f'${val * 100_000 / 1e6:.1f}M'))
    cb.set_label('Price (in millions)', fontsize=12, labelpad=10)
    ax.view_init(elev=40, azim=165)
    fig.tight_layout()
    fig.savefig('figures/nyc_housing_price_3d.png', dpi=300)
    plt.close()
# 3D Location Plot:1 ends here

# [[file:../code.org::*Hypothesis Testing][Hypothesis Testing:1]]
def test_hypothesis():
    """Test if average price is significantly higher than $1M"""
    from scipy import stats

    df = pd.read_csv('NY-House-Dataset_Filtered.csv')
    prices = df['PRICE']
    mu_0 = 1_000_000  # Null hypothesis value

    sample_mean = np.mean(prices)
    sample_std = np.std(prices, ddof=1)
    n = len(prices)
    z_score = (sample_mean - mu_0) / (sample_std / np.sqrt(n))
    p_value = 1 - stats.norm.cdf(z_score)

    print("\n=== Hypothesis Test Results ===")
    print(f"Sample Mean: ${sample_mean:,.2f}")
    print(f"Z-Score: {z_score:.4f}")
    print(f"P-Value (one-tailed): {p_value:.6f}")

    alpha = 0.05
    if p_value < alpha:
        print("Conclusion: Reject null - average price > $1M")
    else:
        print("Conclusion: Fail to reject null")
# Hypothesis Testing:1 ends here

# [[file:../code.org::*Confidence Interval][Confidence Interval:1]]
def calculate_confidence_interval():
    """Calculate and plot 95% CI for mean price"""
    import matplotlib.pyplot as plt
    from scipy import stats

    df = pd.read_csv('NY-House-Dataset_Filtered.csv')
    price_millions = df['PRICE'] / 1_000_000

    sample_mean = np.mean(price_millions)
    std_dev = np.std(price_millions, ddof=1)
    sample_size = len(price_millions)
    error = std_dev / np.sqrt(sample_size)
    ci = stats.t.interval(0.95, sample_size-1, sample_mean, error)

    print("\n=== Confidence Interval ===")
    print(f"95% CI: [${ci[0]:,.3f}M , ${ci[1]:,.3f}M]")
    print(f"Sample Mean: ${sample_mean:,.3f}M")
    print(f"Skewness: {stats.skew(price_millions):.3f}")

    # Plot
    plt.figure(figsize=(8, 2))
    plt.errorbar(x=[sample_mean], y=[1],
                xerr=[[sample_mean - ci[0]], [ci[1] - sample_mean]],
                fmt='o', color='navy', ecolor='skyblue', capsize=8)
    plt.xlim(min(1.7, ci[0] - 0.05), max(2.1, ci[1] + 0.05))
    plt.yticks([])
    plt.xlabel("Housing Price (in millions USD)")
    plt.title("95% Confidence Interval for NYC Mean Housing Price (2024)")
    plt.text(sample_mean, 1.05, f"Mean: ${sample_mean:.3f}M",
            ha='center', va='top', fontsize=12, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/nyc_housing_confidence_interval.png", dpi=300)
    plt.close()
# Confidence Interval:1 ends here

# [[file:../code.org::*Regression Analysis][Regression Analysis:1]]
def run_regressions():
    """Run linear regressions for price predictors"""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    import os
    import matplotlib.pyplot as plt

    df = pd.read_csv("NY-House-Dataset_Filtered.csv")
    features = ['PROPERTYSQFT', 'BEDS', 'BATH']

    print("\n=== Regression Results ===")
    for feature in features:
        X = df[[feature]]
        y = df['PRICE']

        model = LinearRegression()
        model.fit(X, y)
        y_hat = model.predict(X)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X, y, color='lightblue', alpha=0.6)
        plt.plot(X, y_hat, color='red', linewidth=2)
        plt.xlabel(feature)
        plt.ylabel("Price (USD)")
        plt.grid(True)
        plt.savefig(f"figures/price_vs_{feature.lower()}.png", dpi=300)
        plt.close()

        # Print results
        print(f"{feature}:")
        print(f"  Intercept: ${model.intercept_:,.2f}")
        print(f"  Slope: {model.coef_[0]:,.2f}")
        print(f"  r²: {r2_score(y, y_hat):.3f}\n")
# Regression Analysis:1 ends here

# [[file:../code.org::*Main Execution][Main Execution:1]]
if __name__ == "__main__":
    trim_data()
    configure_plots()
    summarize_data()
    plot_price_distribution()
    plot_3d_location()
    test_hypothesis()
    calculate_confidence_interval()
    run_regressions()
# Main Execution:1 ends here
