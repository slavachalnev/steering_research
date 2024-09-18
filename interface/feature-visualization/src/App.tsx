import { useState } from "react";
import "./App.css";
import FeatureColumn from "./components/FeatureColumn";

import { BORDER_RADIUS, FONT_SIZE, PADDING } from "./utils";
import bret_activations from "./data/base_bret_activations_small.json";
import Inspector from "./components/Inspector";
import { ProcessedFeaturesType } from "./types";

export default function App() {
	const [uniqueFeatures, setUniqueFeatures] =
		useState<ProcessedFeaturesType[]>(bret_activations);
	const [processedFeatures, setProcessedFeatures] =
		useState<ProcessedFeaturesType[]>(bret_activations);

	const [magnified, setMagnified] = useState<number>(-1);

	const onMagnify = (id: string) => {
		const index: number = processedFeatures.findIndex(
			(feature) => feature.id === id
		);
		if (magnified == processedFeatures[index].feature) {
			setMagnified(-1);
		} else {
			setMagnified(processedFeatures[index].feature);
		}
	};

	// const removeFeature = (id: string) => {
	// 	const indexToRemove = processedFeatures.findIndex(
	// 		(feature) => feature.id == id
	// 	);

	// 	if (indexToRemove !== -1) {
	// 		setProcessedFeatures((prevFeatures) =>
	// 			prevFeatures.filter((_, index) => index !== indexToRemove)
	// 		);
	// 		setActivations((prevActivations) =>
	// 			prevActivations.map((row) =>
	// 				row.filter((_, index) => index !== indexToRemove)
	// 			)
	// 		);

	// 		if (magnified == processedFeatures[indexToRemove].feature) {
	// 			setMagnified(-1);
	// 		}
	// 	}
	// };

	return (
		<div
			style={{
				display: "flex",
				flexDirection: "column",
				alignItems: "center",
				width: "100vw",
				fontSize: FONT_SIZE,
				borderRadius: BORDER_RADIUS,
				padding: PADDING,
			}}
		>
			<Inspector
				processedFeatures={processedFeatures}
				setProcessedFeatures={setProcessedFeatures}
				uniqueFeatures={uniqueFeatures}
				setUniqueFeatures={setUniqueFeatures}
				setMagnified={setMagnified}
				magnified={magnified}
			/>
			{processedFeatures.length > 0 && (
				<FeatureColumn
					onMagnify={onMagnify}
					processedFeatures={processedFeatures}
					magnified={magnified}
					// removeFeature={removeFeature}
				/>
			)}
		</div>
	);
}
