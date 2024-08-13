import React, { useState, useEffect, useMemo, useRef } from "react";
import "./App.css";
import { fetchExpandedCluster } from "./utils";
import html2canvas from "html2canvas";

interface From {
	feature: number;
	value: number;
}

// Add this interface near the top of your file, before the App component
interface Item {
	feature: number;
	from: From | null;
}

const gemma2_2b_url = "gemma-2-2b/12-gemmascope-res-16k";
const gemma_2b_url = "gemma-2b/6-res-jb";

// Add this new component
const CachedIframe = React.memo(({ feature }: { feature: number }) => {
	const src = useMemo(
		() => `https://neuronpedia.org/${gemma_2b_url}/${feature}?embed=true`,
		[feature]
	);

	return (
		<iframe
			src={src}
			title="Neuronpedia"
			className="neuronpedia-iframe"
			style={{
				pointerEvents: "auto",
				overflow: "auto",
				width: "400px",
				height: "600px",
				border: "1px solid lightgrey",
				boxShadow: "0 0 10px 0 rgba(0, 0, 0, 0.1)",
			}}
		/>
	);
});

function App() {
	const [inputText, setInputText] = useState("");
	const [cluster, setCluster] = useState<Item[]>([]);
	const [selectedItem, setSelectedItem] = useState<Item | null>(null);
	const [frontier, setFrontier] = useState<Item[]>([]);
	const [threshold, setThreshold] = useState<number>(0.5);
	const clusterRef = useRef<HTMLDivElement>(null);
	const [bulkInput, setBulkInput] = useState("");

	const addToCluster = (itemToAdd: Item) => {
		const newCluster = cluster.some(
			(item) => item.feature === itemToAdd.feature
		)
			? cluster
			: [itemToAdd, ...cluster];
		setCluster(newCluster);
	};

	const findFrontier = async () => {
		if (selectedItem === null) {
			return;
		}

		const newCluster = cluster.some(
			(item) => item.feature === selectedItem.feature
		)
			? cluster
			: [
					selectedItem,
					...cluster.filter((item) => item.feature !== selectedItem.feature),
			  ];

		const expandedCluster = await fetchExpandedCluster(
			selectedItem.feature,
			newCluster.map((item) => item.feature),
			threshold
		);

		// Update frontier with proper Item objects
		const updatedFrontier = expandedCluster.new_frontier.map(
			(feature: number, index: number) => ({
				feature,
				from: {
					feature: selectedItem.feature,
					value: expandedCluster.frontier_similarities[index],
				},
			})
		);

		setCluster(newCluster);
		setFrontier(updatedFrontier);
	};

	const updateSelectedItem = (node: number | Item) => {
		if (typeof node === "number") {
			const newItem: Item = {
				feature: node,
				from: null,
			};
			setSelectedItem(newItem);
			// setInputText(node.toString());
		} else {
			setSelectedItem(node);
			// setInputText(node.feature.toString());
		}
	};

	const copyClusterToClipboard = () => {
		const clusterFeatures = cluster.map((item) => item.feature).join(", ");
		navigator.clipboard
			.writeText(clusterFeatures)
			.then(() => {
				alert("Cluster features copied to clipboard!");
			})
			.catch((err) => {
				console.error("Failed to copy: ", err);
			});
	};

	const captureClusterAsPNG = () => {
		if (clusterRef.current) {
			html2canvas(clusterRef.current).then((canvas) => {
				const image = canvas
					.toDataURL("image/png")
					.replace("image/png", "image/octet-stream");
				const link = document.createElement("a");
				link.download = "cluster.png";
				link.href = image;
				link.click();
			});
		}
	};

	const addBulkItems = () => {
		const items = bulkInput.split(",").map((item) => item.trim());
		const newItems = items.map((item) => ({
			feature: parseInt(item),
			from: null,
		}));
		setCluster((prevCluster) => {
			const uniqueNewItems = newItems.filter(
				(newItem) =>
					!prevCluster.some(
						(existingItem) => existingItem.feature === newItem.feature
					)
			);
			return [...uniqueNewItems, ...prevCluster];
		});
		setBulkInput("");
	};

	const deleteFromCluster = (featureToDelete: number) => {
		setCluster((prevCluster) =>
			prevCluster.filter((item) => item.feature !== featureToDelete)
		);
	};

	useEffect(() => {
		findFrontier();
	}, [threshold, selectedItem]);

	useEffect(() => {
		console.log("cluster");
		console.log(cluster);
	}, [cluster]);

	useEffect(() => {
		console.log("frontier");
		console.log(frontier);
	}, [frontier]);

	return (
		<div>
			<div>
				<h3>Cluster</h3>

				<button onClick={copyClusterToClipboard}>Copy Cluster Features</button>
				<button onClick={captureClusterAsPNG}>Capture Cluster as PNG</button>
				<div>
					<input
						type="text"
						value={bulkInput}
						onChange={(e) => setBulkInput(e.target.value)}
						placeholder="Enter comma-separated features (e.g., 12665, 4343, 7134)"
					/>
					<button onClick={addBulkItems}>Add Bulk Items</button>
				</div>
				<div className="row" ref={clusterRef}>
					{cluster.map((item: Item) => (
						<div key={item.feature}>
							<div>
								{item.feature}
								<button onClick={() => updateSelectedItem(item)}>
									Search Frontier
								</button>
								<button onClick={() => deleteFromCluster(item.feature)}>
									Delete
								</button>
							</div>
							<CachedIframe feature={item.feature} />
						</div>
					))}
				</div>
			</div>
			<div>
				<h3>Frontier</h3>
				<input
					type="text"
					value={inputText}
					onChange={(e) => setInputText(e.target.value)}
					placeholder="Enter text here"
				/>
				<button onClick={() => updateSelectedItem(Number(inputText))}>
					Expand
				</button>

				<div>
					<label htmlFor="threshold">Threshold: {threshold.toFixed(2)}</label>
					<input
						type="range"
						id="threshold"
						min="0"
						max="1"
						step="0.01"
						value={threshold}
						onChange={(e) => setThreshold(parseFloat(e.target.value))}
					/>
				</div>

				<div className="row">
					{frontier
						.sort((a, b) => {
							// Sort in descending order of value score
							return (b.from?.value ?? 0) - (a.from?.value ?? 0);
						})
						.filter((item) => !cluster.includes(item))
						.slice(0, 10)
						.map((item: Item) => (
							<div key={item.feature}>
								<div>
									{item.feature}
									{item.from && (
										<span> (Score: {item.from.value.toFixed(3)})</span>
									)}
									<button onClick={() => addToCluster(item)}>
										Add to cluster
									</button>
								</div>
								<CachedIframe feature={item.feature} />
							</div>
						))}
				</div>
			</div>
		</div>
	);
}

export default App;
