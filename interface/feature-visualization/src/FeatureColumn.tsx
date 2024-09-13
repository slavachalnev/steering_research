import React, { useState, useRef, useEffect } from "react";
import FeatureCard from "./FeatureCard";
import data from "./base_bret_activations.json";
import { getBaseUrl } from "./utils";

interface FeatureData {
	binMax: number;
	binMin: number;
	maxValue: number;
	minValue: number;
	tokens: string[];
	values: number[];
}

interface ProcessedFeaturesType {
	feature: number;
	id: string;
	feature_results: FeatureData[];
}

const FeatureColumn = ({
	processedFeatures,
	setProcessedFeatures,
	onMagnify,
}: {
	processedFeatures: ProcessedFeaturesType[];
	setProcessedFeatures: (features: ProcessedFeaturesType[]) => void;
	onMagnify: (id: string) => void;
}) => {
	const [newFeature, setNewFeature] = useState<string>("");
	const [isInputVisible, setIsInputVisible] = useState(false);
	const inputRef = useRef<HTMLInputElement>(null);
	// const [searchResults, setSearchResults] = useState<SearchResult[]>([]);

	// const [processedFeatures, setProcessedFeatures] = useState<
	// 	ProcessedFeaturesType[]
	// >([]);

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
		setProcessedFeatures(
			processedFeatures.filter((feature) => feature.id !== id)
		);
	};

	const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		const value = e.target.value;
		setNewFeature(value);
		// debouncedSearch(value);
	};

	const handleAddFeature = (e: React.KeyboardEvent<HTMLInputElement>) => {
		// if (e.key === "Enter" && newFeature.trim() !== "") {
		// 	const featureNumber = parseInt(newFeature);
		// 	if (!isNaN(featureNumber)) {
		// 		setFeatures([...features, { id: crypto.randomUUID(), featureNumber }]);
		// 		setNewFeature("");
		// 		collapseInput();
		// 	}
		// }
	};

	const expandInput = () => {
		setIsInputVisible(true);
	};

	const collapseInput = () => {
		setIsInputVisible(false);
		setNewFeature("");
	};

	const getAllFeatureData = async () => {
		// const url = `${getBaseUrl()}/get_feature?features=${features
		// 	.map((feature) => feature.featureNumber)
		// 	.join(",")}`;
		// const url = `${getBaseUrl()}/get_feature?features=7914, 7850, 15370, 5365, 1990, 16339, 12634, 3288, 9758, 7854, 4291, 1633, 13425, 15710, 3118, 628, 8570, 8318, 5315, 3969`;
		// const response = await fetch(url, {
		// 	method: "GET",
		// 	headers: {
		// 		"Content-Type": "application/json",
		// 	},
		// });
		// const data = await response.json();

		const dataWithId: ProcessedFeaturesType[] = data.map((d: any) => {
			return Object.assign(d, {
				id: crypto.randomUUID(),
			});
		});
		setProcessedFeatures([...processedFeatures, ...dataWithId]);
	};

	// useEffect(() => {
	// 	getAllFeatureData();
	// }, []);

	return (
		<div
			style={{
				height: "70vh",
				display: "flex",
				flexDirection: "column",
				overflow: "hidden",
			}}
		>
			<div
				style={{
					display: "flex",
					flexDirection: "column",
					// alignItems: "center",
					transition: "width 0.3s",
					// height: "auto",
					flexGrow: 1,
					overflowY: "auto",
				}}
			>
				<div style={{ display: "flex", flexDirection: "column", gap: ".5px" }}>
					{processedFeatures.map(
						(feature: ProcessedFeaturesType, i: number) => {
							return (
								<React.Fragment key={feature.id + i}>
									<FeatureCard
										feature={feature.feature}
										featureId={feature.id}
										onDelete={removeFeature}
										onMagnify={onMagnify}
										activations={feature.feature_results}
									/>
									<div style={{ height: "10px", width: "100%" }} />
								</React.Fragment>
							);
						}
					)}
				</div>

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
			</div>
		</div>
	);
};

export default FeatureColumn;
