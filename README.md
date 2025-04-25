
# 🌐 Network Analysis of India’s Road Network for Optimising Supply Chain

This project applies **network science** techniques to analyze India's road infrastructure and enhance the **efficiency of supply chain distribution**. By leveraging centrality measures, structural analysis, and optimization strategies, we identify critical nodes and potential improvements to road networks.

---

## 🧭 Project Structure

```
.
├── highway/                      # Focused analysis on national highways for logistics optimization
├── one_road_one_node_analysis/  # Simplified graph where each road is modeled with a unique node
├── road_network_analysis/       # General graph analytics on India's road network
├── README.md                    # Project overview and guide
```

---

## 🎯 Objectives

- Construct spatial graphs from India's road network data (GeoJSON).
- Identify **critical roads** and **vulnerable intersections** using centrality metrics.
- Simulate and analyze **edge additions** to enhance network robustness.
- Explore how simplified modeling affects interpretability and results.

---

## 🧪 Methodologies Used

- 📍 **Graph Construction**: Using geospatial data to define nodes (intersections) and edges (roads).
- 🔄 **Centrality Measures**: Betweenness, stress centrality, closeness, and more.
- ➕ **Edge Addition Heuristics**: Identify links that improve flow and reduce stress.
- ⚙️ **Parallel Processing**: Speed up large-graph experiments using multiprocessing.

---

## 📂 Subfolders Explained

### `highway/`
Targeted study of India's highway system. Focuses on finding bottlenecks and potential improvements for long-distance logistics.

### `one_road_one_node_analysis/`
Experimental setup where each road is represented by a single node. Useful for understanding alternate modeling strategies.

### `road_network_analysis/`
Original and broad-scale graph analysis of India's road infrastructure, including preprocessing, metrics, and exploratory insights.

---

## 📈 Results & Insights

- Identified **key chokepoints** in the highway system.
- Demonstrated improvement in **maximum stress centrality** through heuristic edge addition.
- Comparative analysis of modeling strategies (`highway` vs `one_road_one_node`).

---

## 🔧 Requirements

- Python ≥ 3.8
- Libraries: `networkx`, `geopandas`, `matplotlib`, `numpy`, `shapely`, `tqdm`

```bash
pip install -r requirements.txt
```

---

## 🙌 Contributors

Project by students of **IIT Bhilai**, as part of the Network Science coursework and research.

---

## 📜 License

For academic and research purposes only. For inquiries or collaborations, please contact the maintainers.

---

## 📌 Acknowledgements

- Indian Government Open Data Platform for GeoJSON road datasets.
- NetworkX and GeoPandas for providing robust graph and geospatial tooling.
