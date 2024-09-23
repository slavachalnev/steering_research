import { useState } from "react";
import "./App.css";
import FeatureColumn from "./components/FeatureColumn";

import { BORDER_RADIUS, FONT_SIZE, PADDING } from "./utils";
import bret_activations from "./data/base_bret_activations_small.json";
import Inspector from "./components/Inspector";
import { ProcessedFeaturesType } from "./types";
import Explorer from "./components/Explorer";

export default function App() {
	const [uniqueFeatures, setUniqueFeatures] =
		useState<ProcessedFeaturesType[]>(bret_activations);
	const [processedFeatures, setProcessedFeatures] =
		useState<ProcessedFeaturesType[]>(bret_activations);

	const [magnified, setMagnified] = useState<number | null>(null);

	const [mode, setMode] = useState<"inspector" | "explorer">("inspector");

	const onMagnify = (id: string) => {
		const index: number = processedFeatures.findIndex(
			(feature) => feature.id === id
		);
		if (magnified == processedFeatures[index].feature) {
			setMagnified(null);
		} else {
			setMagnified(processedFeatures[index].feature);
		}
	};

	const toggleMode = () => {
		if (mode === "inspector") {
			setMagnified(-1);
			setProcessedFeatures([]);
			setMode("explorer");
		} else {
			setProcessedFeatures(uniqueFeatures);
			setMode("inspector");
		}
	};

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
			<button
				style={{
					backgroundColor: "white",
					width: "fit-content",
					border: "none",
					color: "grey",
					textAlign: "center",
					textDecoration: "none",
					display: "inline-block",
					margin: "4px 2px",
					cursor: "pointer",
					whiteSpace: "nowrap",
					borderRadius: BORDER_RADIUS,
					fontSize: FONT_SIZE,
					padding: PADDING,
				}}
				onClick={toggleMode}
			>
				Switch to {mode === "inspector" ? "Explorer" : "Inspector"} Mode
			</button>
			{mode === "inspector" ? (
				<Inspector
					processedFeatures={processedFeatures}
					setProcessedFeatures={setProcessedFeatures}
					uniqueFeatures={uniqueFeatures}
					setUniqueFeatures={setUniqueFeatures}
					setMagnified={setMagnified}
					magnified={magnified}
				/>
			) : (
				<Explorer setProcessedFeatures={setProcessedFeatures} />
			)}
			{processedFeatures.length > 0 && (
				<FeatureColumn
					onMagnify={mode === "inspector" ? onMagnify : undefined}
					processedFeatures={processedFeatures}
					magnified={magnified}
				/>
			)}
		</div>
	);
}
