import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.cm import ScalarMappable
import pandas as pd

def compute_permeability_matrix(phi_x, phi_y, phi_z, Bx, By, Bz, mx, my, mz):
    k11 = Bx * (phi_x)**mx
    k22 = By * (phi_y)**my
    k33 = Bz * (phi_z)**mz

    permeability_matrix = np.maximum(np.array([[k11, 0, 0],
                                               [0, k22, 0],
                                               [0, 0, k33]]), 0.0)

    return permeability_matrix

def sph2cart(r, phi, tta):
    x = r * np.sin(tta) * np.cos(phi)
    y = r * np.sin(tta) * np.sin(phi)
    z = r * np.cos(tta)
    return x, y, z
 
def ellips2cart(r, phi, tta, a, b, c):
    x = a * r * np.sin(tta) * np.cos(phi)
    y = b * r * np.sin(tta) * np.sin(phi)
    z = c * r * np.cos(tta)
    return x, y, z    

# Streamlit interface
st.title("Orthotropic Permeability and Porous Resistivity Tensor Visualization")

# User input for porosity values
phi_x = st.sidebar.number_input("Porosity in the x-axis", min_value=0.0, max_value=1.0, step=0.01, value=0.2)
phi_y = st.sidebar.number_input("Porosity in the y-axis", min_value=0.0, max_value=1.0, step=0.01, value=0.3)
phi_z = st.sidebar.number_input("Porosity in the z-axis", min_value=0.0, max_value=1.0, step=0.01, value=0.4)

# User input for base permeability constant and exponent
Bx = st.sidebar.number_input("Base Permeability Constant in x direction", min_value=1.0E-11, value=1.0E-9, format='%.2e', step=1.0E-9)
By = st.sidebar.number_input("Base Permeability Constant in y direction", min_value=1.0E-11, value=1.0E-9, format='%.2e', step=1.0E-9)
Bz = st.sidebar.number_input("Base Permeability Constant in z direction", min_value=1.0E-11, value=1.0E-9, format='%.2e', step=1.0E-9)
mx = st.sidebar.number_input("Exponent in x", min_value=0.0, value=2.0)
my = st.sidebar.number_input("Exponent in y", min_value=0.0, value=2.0)
mz = st.sidebar.number_input("Exponent in z", min_value=0.0, value=2.0)

# Compute permeability matrix
permeability_matrix = compute_permeability_matrix(phi_x, phi_y, phi_z, Bx, By, Bz, mx, my, mz)

# Calculate the porous resistivity as inverse of the permeability matrix
porous_resistivity_matrix = np.linalg.inv(permeability_matrix)

# Calculate the property values at each point
theta = np.linspace(0, np.pi, 200)
phi = np.linspace(0, 2 * np.pi, 200)
Theta, Phi = np.meshgrid(theta, phi)

# Streamlit sidebar options
visualization_option = st.sidebar.radio("Select Visualization", ('Permeability', 'Porous Resistivity'))
#cmap_name = st.sidebar.selectbox("Select a color map", plt.colormaps()) # perhaps this will set magma

# Download permeability matrix as CSV
if st.button("Download Permeability Matrix"):
    df_permeability = pd.DataFrame(permeability_matrix)
    df_permeability.to_csv("permeability_matrix.csv", index=False)
    st.success("Permeability Matrix downloaded successfully!")

# Download porous resistivity matrix as CSV
if st.button("Download Porous Resistivity Matrix"):
    df_porosity_resistivity = pd.DataFrame(porous_resistivity_matrix)
    df_porosity_resistivity.to_csv("porous_resistivity_matrix.csv", index=False)
    st.success("Porous Resistivity Matrix downloaded successfully!")

if visualization_option == 'Permeability':
    # Calculate the major and minor radii of the ellipsoid based on permeability values
    ap = np.sqrt(permeability_matrix[0, 0] / Bx)
    bp = np.sqrt(permeability_matrix[1, 1] / By)
    cp = np.sqrt(permeability_matrix[2, 2] / Bz)
    X, Y, Z = ellips2cart(1, Phi, Theta, ap, bp, cp)

    # Calculate the permeability tensor values
    tensor_values = (
        permeability_matrix[0, 0] * (X/ap)**2
        + permeability_matrix[1, 1] * (Y/bp)**2
        + permeability_matrix[2, 2] * (1 - (X/ap)**2 - (Y/bp)**2)
    )

    title = 'Orthotropic Permeability Tensor'
    zlabel = 'Permeability (m$^2$)'

elif visualization_option == 'Porous Resistivity':
    # Calculate the major and minor radii of the ellipsoid based on porous resistivity values
    ar = np.sqrt(porous_resistivity_matrix[0,0] * Bx)
    br = np.sqrt(porous_resistivity_matrix[1, 1] * By)
    cr = np.sqrt(porous_resistivity_matrix[2, 2] * Bz)
    X, Y, Z = ellips2cart(1, Phi, Theta, ar, br, cr)

    # Calculate the porous resistivity tensor values
    tensor_values = (
        porous_resistivity_matrix[0, 0] * (X/ar)**2
        + porous_resistivity_matrix[1, 1] * (Y/br)**2
        + porous_resistivity_matrix[2, 2] * (1 - (X/ar)**2 - (Y/br)**2)
    )

    title = 'Porous Resistivity Tensor'
    zlabel = 'Porous Resistivity (m$^{-2}$)'
# Plot porous resistivity tensor
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

cmap_name = st.sidebar.selectbox("Select a color map", ['jet', 'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                      'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
                      'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 
                      'turbo', 'nipy_spectral', 'gist_ncar', 'Pastel1', 'Pastel2',
                      'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10',
                      'tab20', 'tab20b', 'tab20c', 'twilight', 'twilight_shifted',
                      'hsv', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                      'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'binary',
                      'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring',
                      'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot',
                      'gist_heat', 'copper', 'Greys', 'Purples', 'Blues', 'Greens',
                      'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu',
                      'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                      'viridis', 'plasma', 'inferno', 'magma', 'cividis'])

# Set the colormap and normalization
cmap = cm.get_cmap(cmap_name)
norm = plt.Normalize(0, np.max(tensor_values))

#ax.plot_surface(
#    X, Y, Z, facecolors=plt.get_cmap(cmap)(norm(tensor_values)),
#    rstride=1, cstride=1, linewidth=0.1, antialiased=True
#)
# Create a ScalarMappable object
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(tensor_values)

# Plot the surface with colormap
ax.plot_surface(
    X, Y, Z, facecolors=sm.to_rgba(tensor_values),
    rstride=1, cstride=1, linewidth=0.1, antialiased=True
)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(title)
###########################################################
#For controlling the distance between the labels in axes
# Adjust spacing and rotation of axes labels
#ax.xaxis.set_rotate_label(True)
#ax.yaxis.set_rotate_label(True)
#ax.zaxis.set_rotate_label(False)
#ax.tick_params(axis='x', pad=1)
#ax.tick_params(axis='y', pad=1)
#ax.tick_params(axis='z', pad=1)
#########################################################
#Code to make something look elliptic
ax.set_box_aspect([np.ptp(X), np.ptp(Y), np.ptp(Z)])
#########################################################
#Distance between the figure and the colorbar
# Adjust distance between figure and color scale bar
fig.tight_layout(pad=5)

# Add the colorbar
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), pad=0.1)
cbar.set_label(zlabel, fontsize=12)
cbar.ax.tick_params(labelsize=10)

st.pyplot(fig)


