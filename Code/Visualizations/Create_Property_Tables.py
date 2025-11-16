"""Create publication-quality property tables for both datasets."""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Property descriptions: (Dataset Name, Astrophysics Name, Range/Cut, Description)
ALFALFA_PROPS = {
    "BARYONIC_MASS": ("BARYONIC_MASS", "Baryonic Mass", "10⁶ - 10¹² M_☉", "Total baryonic mass (stars + 1.4×HI gas)"),
    "COLOR_U_R": ("COLOR_U_R", "Optical Color (u-r)", "-0.5 to 4.0 mag", "u-band minus r-band color index"),
    "ELPETRO_B300": ("ELPETRO_B300", "Star Formation Rate", "10⁻⁸ to 10", "B300 parameter from SED fitting"),
    "SERSIC_N": ("SERSIC_N", "Sersic Index", "0.0 to 6.0", "Light profile concentration parameter"),
    "ELPETRO_METS": ("ELPETRO_METS", "Stellar Metallicity", "-2.5 to 0.5 dex", "Stellar metallicity [Z/H]"),
    "ELPETRO_MTOL": ("ELPETRO_MTOL", "Mass-to-Light Ratio", "0.1 to 10.0 M_☉/L_☉", "Stellar mass-to-light ratio"),
    "ELPETRO_BA": ("ELPETRO_BA", "Axis Ratio", "0.0 to 1.0", "Minor-to-major axis ratio (b/a)"),
    "ELPETRO_TH50_R": ("ELPETRO_TH50_R", "Half-Light Radius", "0.0 to 25.0 arcsec", "Effective radius containing 50% of light"),
    "ZDIST": ("ZDIST", "Redshift", "0.0 to 0.15", "Cosmological redshift"),
    "logMH": ("logMH", "HI Mass", "6.0 to 10.5 log(M_☉)", "Neutral atomic hydrogen mass"),
    "ELPETRO_MASS": ("ELPETRO_MASS", "Stellar Mass", "10⁶ - 10¹² M_☉", "Total stellar mass"),
    "ELPETRO_ABSMAG_R": ("ELPETRO_ABSMAG_R", "Absolute Magnitude (r)", "-25.0 to -10.0 mag", "r-band absolute magnitude"),
    "W50": ("W50", "HI Velocity Width", "20 to 500 km/s", "HI velocity width at 50% peak flux"),
}

NSA_PROPS = {
    "COLOR_U_R": ("COLOR_U_R", "Optical Color (u-r)", "-0.5 to 4.0 mag", "u-band minus r-band color index"),
    "ELPETRO_B300": ("ELPETRO_B300", "Star Formation Rate", "10⁻⁸ to 10", "B300 parameter from SED fitting"),
    "SERSIC_N": ("SERSIC_N", "Sersic Index", "0.0 to 6.0", "Light profile concentration parameter"),
    "ELPETRO_METS": ("ELPETRO_METS", "Stellar Metallicity", "-2.5 to 0.5 dex", "Stellar metallicity [Z/H]"),
    "ELPETRO_MTOL": ("ELPETRO_MTOL", "Mass-to-Light Ratio", "0.1 to 10.0 M_☉/L_☉", "Stellar mass-to-light ratio"),
    "ELPETRO_BA": ("ELPETRO_BA", "Axis Ratio", "0.0 to 1.0", "Minor-to-major axis ratio (b/a)"),
    "ELPETRO_TH50_R": ("ELPETRO_TH50_R", "Half-Light Radius", "0.0 to 25.0 arcsec", "Effective radius containing 50% of light"),
    "ZDIST": ("ZDIST", "Redshift", "0.0 to 0.15", "Cosmological redshift"),
    "ELPETRO_MASS": ("ELPETRO_MASS", "Stellar Mass", "10⁶ - 10¹² M_☉", "Total stellar mass"),
    "ELPETRO_ABSMAG_R": ("ELPETRO_ABSMAG_R", "Absolute Magnitude (r)", "-25.0 to -10.0 mag", "r-band absolute magnitude"),
}


def create_table(data_path, props_dict, dataset_name, output_path):
    with open(data_path, "rb") as f:
        data_dict = pickle.load(f)
    
    n_galaxies = len(next(iter(data_dict.values())))
    
    fig, ax = plt.subplots(figsize=(14, len(props_dict) * 0.6 + 1))
    ax.axis("off")
    
    table_data = [["Property", "Physical Quantity", "Range/Cut", "Description"]]
    
    for key in props_dict.keys():
        dataset_name, physical_quantity, range_cut, desc = props_dict[key]
        table_data.append([dataset_name, physical_quantity, range_cut, desc])
    
    table = ax.table(
        cellText=table_data,
        cellLoc="left",
        loc="center",
        colWidths=[0.25, 0.25, 0.20, 0.30],
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor("#34495e")
        cell.set_text_props(weight="bold", color="white", fontsize=11)
        cell.set_edgecolor("white")
        cell.set_linewidth(1.5)
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(4):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor("#ecf0f1")
            else:
                cell.set_facecolor("white")
            cell.set_edgecolor("#bdc3c7")
            cell.set_linewidth(1)
    
    ax.set_title(
        f"{dataset_name} Dataset Properties\n(N = {n_galaxies:,} galaxies)",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


# Create tables
create_table(
    os.path.join(REPO_ROOT, "Data", "alfalfa_nsa_final_13props.pkl"),
    ALFALFA_PROPS,
    "ALFALFA × NSA",
    os.path.join(REPO_ROOT, "Plots", "Tables", "alfalfa_properties_table.png"),
)

create_table(
    os.path.join(REPO_ROOT, "Data", "nsa_final_10props.pkl"),
    NSA_PROPS,
    "NSA-only",
    os.path.join(REPO_ROOT, "Plots", "Tables", "nsa_properties_table.png"),
)

print("\nProperty tables created successfully!")

