import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mc

# ==========================================
# 1. DATA SETUP
# ==========================================
full_name_mapping = {
    "PER": "Person", "TTL": "Title", "NIC": "Nickname", "PSE": "Pseudonym", "ROLE": "Role",
    "NAT": "Nationality", "REL": "Religion", "POL": "Political Group", "ETH": "Ethnicity",
    "LAT": "Latitude", "LON": "Longitude", "CON": "Continent", "COU": "Country", "STA": "State",
    "PRO": "Province","CIT": "City", "DIS": "District", "SUB": "Sub-district", "VIL": "Village", "ROA": "Road",
    "ISL": "Island", "RIV": "River", "MOU": "Mountain", "OCE": "Ocean", "NPL": "Natural Place",
    "RPL": "Religious Place", "NPK": "National Park", "EDU": "Education Inst.",
    "LOC": "Location", "ODY": "Dynasty", "STY": "State Agency", "EVH": "Hist. Event",
    "EVN": "Natural Event", "EVO": "Other Event", "DATE": "Date", "ERA": "Era/Period",
    "DUR": "Duration", "SEA": "Season", "NUM": "Number", "MEA": "Measurement", "GOD": "God",
    "ANI": "Animal", "LAN": "Language", "ANT": "Antique", "ART": "Artifact", "WEA": "Weapon",
    "VEH": "Vehicle", "PRT": "Product", "FOO": "Food", "DSE": "Disease", "MAT": "Material",
    "DES": "Designation", "DTM": "Date/Time", "O": "O", "ORG":"Organization"
}

main_cat_mapping = {
    'PER': 'PERSON', 'NORP': 'NORP', 'GPE': 'GPE', 'LOC': 'LOCATION',
    'ORG': 'ORGANIZATION', 'EVENT': 'EVENT', 'TIME': 'TIME',
    'NUM': 'NUMBER', 'MISC': 'MISC'
}

data_structure = {
    main_cat_mapping['PER']: ['PER', 'TTL', 'NIC', 'PSE', 'ROLE'],
    main_cat_mapping['NORP']: ['NAT', 'REL', 'POL', 'ETH'],
    main_cat_mapping['GPE']: ['LAT', 'LON', 'CON', 'COU', 'STA', 'PRO', 'DIS', 'SUB', 'VIL', 'ROA', 'CIT'],
    main_cat_mapping['LOC']: ['LOC', 'ISL', 'RIV', 'MOU', 'OCE', 'NPL', 'RPL', 'NPK', 'EDU'],
    main_cat_mapping['ORG']: ['ORG', 'ODY', 'STY'],
    main_cat_mapping['EVENT']: ['EVH', 'EVN', 'EVO'],
    main_cat_mapping['TIME']: ['DATE', 'DTM', 'ERA', 'DUR', 'SEA'],
    main_cat_mapping['NUM']: ['NUM', 'MEA'],
    main_cat_mapping['MISC']: ['GOD', 'ANI', 'LAN', 'ANT', 'ART', 'WEA', 'VEH', 'PRT', 'FOO', 'DSE', 'MAT', 'DES']
}

# ==========================================
# 2. COLOR & STYLE
# ==========================================
cmap = plt.get_cmap("nipy_spectral")
main_colors = [cmap(i) for i in np.linspace(0.05, 0.95, 9)]

inner_labels, inner_sizes, inner_colors = [], [], []
outer_labels, outer_sizes, outer_colors_mapped = [], [], []

for i, (main_cat, sub_keys) in enumerate(data_structure.items()):
    inner_labels.append(main_cat)
    inner_sizes.append(len(sub_keys))
    base_color = main_colors[i]
    inner_colors.append(base_color)
    
    for key in sub_keys:
        outer_labels.append(full_name_mapping.get(key, key))
        outer_sizes.append(1)
        c = mc.to_rgba(base_color)
        tint = 0.2
        outer_colors_mapped.append((c[0]+(1-c[0])*tint, c[1]+(1-c[1])*tint, c[2]+(1-c[2])*tint, c[3]))

# ==========================================
# 3. PLOTTING (EXTREME VISIBILITY MODE)
# ==========================================
# Canvas is 26x26 inches. This allows us to use HUGE fonts.
fig, ax = plt.subplots(figsize=(26, 26)) 
ax.axis('equal')

size_ring = 0.35
radius_inner = 1.0 - size_ring
radius_outer = 1.0

# --- INNER RING ---
patches_inner, texts_inner = ax.pie(inner_sizes, radius=radius_inner, colors=inner_colors, labels=inner_labels,
       labeldistance=0.6, wedgeprops=dict(width=size_ring, edgecolor='white', linewidth=3), rotatelabels=True)

for text in texts_inner:
    text.set_fontsize(32) # Huge inner font
    text.set_fontweight('bold')
    text.set_fontfamily('Times New Roman')
    text.set_color('white')
    text.set_rotation_mode('anchor')
    if 90 < text.get_rotation() % 360 < 270:
        text.set_rotation(text.get_rotation() + 180)

# --- OUTER RING ---
patches_outer, _ = ax.pie(outer_sizes, radius=radius_outer, colors=outer_colors_mapped, labels=['']*len(outer_labels), 
       wedgeprops=dict(width=size_ring, edgecolor='white', linewidth=1.5), startangle=0)

# --- OUTER LABELS (BOLD & HUGE) ---
for i, (wedge, label) in enumerate(zip(patches_outer, outer_labels)):
    angle = (wedge.theta2 + wedge.theta1) / 2.
    rad_angle = np.radians(angle)
    
    # Keep labels tight to the circle to save space
    label_dist = radius_outer + 0.03 
    x = label_dist * np.cos(rad_angle)
    y = label_dist * np.sin(rad_angle)
    
    norm_angle = angle % 360
    if 90 < norm_angle < 270:
        rotation = norm_angle + 180
        ha, va = 'right', 'center'
    else:
        rotation = norm_angle
        ha, va = 'left', 'center'
        
    ax.text(x, y, label, rotation=rotation, ha=ha, va=va, 
            fontsize=40, # <--- EXTREME FONT SIZE
            fontweight='bold', # <--- BOLD to prevent blurring
            fontfamily='Times New Roman', 
            rotation_mode='anchor')

plt.tight_layout()

# ==========================================
# 4. EXPORT
# ==========================================
plt.savefig("ner_chart_extreme.png", format='png', dpi=300, bbox_inches='tight')
plt.show()