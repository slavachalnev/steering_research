import React, { useState, useRef, useEffect } from "react";
import FeatureCard from "./FeatureCard";
import { FeatureItem } from "./types";

interface FeatureColumnProps {
	columnSide: "left" | "right";
	inspectFeature: (feature: FeatureItem) => void;
	inspectedFeature: FeatureItem | null;
	// features: FeatureItem[];
	// removeFeature: (id: string) => void;
	// addFeature: (featureNumber: number) => void;
}

const FeatureColumn: React.FC<FeatureColumnProps> = ({
	columnSide,
	inspectFeature,
	inspectedFeature,
	// features,
	// removeFeature,
	// addFeature,
}) => {
	const [features, setFeatures] = useState<FeatureItem[]>([
		{ id: crypto.randomUUID(), featureNumber: 5990 },
		{ id: crypto.randomUUID(), featureNumber: 10138 },
		{ id: crypto.randomUUID(), featureNumber: 1015 },
		{ id: crypto.randomUUID(), featureNumber: 909 },
	]);
	const [newFeature, setNewFeature] = useState<string>("");
	const [isInputVisible, setIsInputVisible] = useState(false);
	const inputRef = useRef<HTMLInputElement>(null);
	// const [searchResults, setSearchResults] = useState<SearchResult[]>([]);

	useEffect(() => {
		if (columnSide === "left") {
			setFeatures([
				{ id: crypto.randomUUID(), featureNumber: 5990 },
				{ id: crypto.randomUUID(), featureNumber: 10138 },
				{ id: crypto.randomUUID(), featureNumber: 1015 },
				{ id: crypto.randomUUID(), featureNumber: 909 },
			]);
		} else if (columnSide === "right") {
			// TODO: get the effects
			setFeatures([
				{ id: crypto.randomUUID(), featureNumber: 10138 },
				{ id: crypto.randomUUID(), featureNumber: 1015 },
				{ id: crypto.randomUUID(), featureNumber: 909 },
			]);
		}
	}, [columnSide]);

	// const debouncedSearch = debounce(async (query: string) => {
	// 	if (query.trim() === "") {
	// 		setSearchResults([]);
	// 		return;
	// 	}

	// 	try {
	// 		const response = await fetch(
	// 			"http://localhost:5000/api/explanation/search",
	// 			{
	// 				method: "POST",
	// 				headers: {
	// 					"Content-Type": "application/json",
	// 				},
	// 				body: JSON.stringify({
	// 					modelId: "gemma-2b",
	// 					layers: ["6-res-jb"],
	// 					query: query,
	// 				}),
	// 			}
	// 		);

	// 		if (!response.ok) {
	// 			throw new Error("Search failed");
	// 		}

	// 		const data = await response.json();
	// 		console.log(data);
	// 		setSearchResults(data.results || []);
	// 	} catch (error) {
	// 		console.error("Search error:", error);
	// 		setSearchResults([]);
	// 	}
	// }, 300); // 300ms delay

	const removeFeature = (id: string) => {
		setFeatures(features.filter((feature) => feature.id !== id));
	};

	const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		const value = e.target.value;
		setNewFeature(value);
		// debouncedSearch(value);
	};

	const handleAddFeature = (e: React.KeyboardEvent<HTMLInputElement>) => {
		if (e.key === "Enter" && newFeature.trim() !== "") {
			const featureNumber = parseInt(newFeature);
			if (!isNaN(featureNumber)) {
				setFeatures([...features, { id: crypto.randomUUID(), featureNumber }]);
				setNewFeature("");
				collapseInput();
			}
		}
	};

	const expandInput = () => {
		setIsInputVisible(true);
	};

	const collapseInput = () => {
		setIsInputVisible(false);
		setNewFeature("");
		// setSearchResults([]);
	};

	useEffect(() => {
		if (isInputVisible && inputRef.current) {
			inputRef.current.focus();
		}
	}, [isInputVisible]);

	return (
		<div>
			<div
				style={{
					display: "flex",
					flexDirection: "column",
					alignItems: "center",
					transition: "width 0.3s",
					// marginLeft: columnSide === "left" ? "25px" : "0px",
					// width: inspectedFeature ? "50%" : "100%",
				}}
			>
				<div
					style={{
						marginRight: "auto",
						marginLeft: "15px",
					}}
				>
					{columnSide == "right" ? (
						<h2>
							{" "}
							<select
								style={{
									border: "none",
									background: "transparent",
									fontSize: "inherit",
									fontWeight: "inherit",
									color: "inherit",
									appearance: "none",
									cursor: "pointer",
								}}
							>
								<option>Cosine Sim</option>
								<option>Feature Effects</option>
								<option>Similar Feature Effects</option>
							</select>
						</h2>
					) : (
						<h2>Collection</h2>
					)}
				</div>

				{features.map((feature) => (
					<React.Fragment key={feature.id}>
						<FeatureCard
							inspectFeature={inspectFeature}
							inspectedFeature={inspectedFeature}
							feature={feature}
							onDelete={removeFeature}
							columnSide={columnSide}
						/>
						<div style={{ height: "10px", width: "100%" }} />
					</React.Fragment>
				))}
				{columnSide === "left" && (
					<div className="add-feature-container">
						<span
							className={`add-icon ${isInputVisible ? "hidden" : ""}`}
							onClick={expandInput}
						>
							+
						</span>
						<input
							ref={inputRef}
							type="text"
							value={newFeature}
							onChange={handleInputChange}
							onKeyDown={handleAddFeature}
							onBlur={collapseInput}
							placeholder="Enter feature or search"
							className={`feature-input ${isInputVisible ? "visible" : ""}`}
						/>
					</div>
				)}
				{/* {searchResults.length > 0 && (
				<div className="search-results">
					{searchResults.map((result: any) => (
						<div key={result.index} className="search-result-item">
							<span>{result.index}</span>
							<span className="search-result-description">
								{result.description}
							</span>
						</div>
					))}
				</div>
			)} */}
			</div>
		</div>
	);
};

export default FeatureColumn;
