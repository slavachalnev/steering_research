import { useState, useEffect } from "react";
import "./App.css";
// import FeatureCard from "./FeatureCard";
import FeatureColumn from "./FeatureColumn";
import { FeatureItem } from "./types";

export default function App() {
	const [inspectedFeature, setInspectedFeature] = useState<FeatureItem | null>(
		null
	);

	const inspectFeature = (feature: FeatureItem) => {
		if (inspectedFeature?.id === feature.id) {
			setInspectedFeature(null);
		} else {
			setInspectedFeature(feature);
		}
	};

	useEffect(() => {
		console.log(inspectedFeature);
	}, [inspectedFeature]);

	return (
		<div
			style={{
				display: "flex",
				// justifyContent: "space-between",
				// transform: inspectedFeature
				// 	? "translateX(0)"
				// 	: "translateX(calc(50% - 50%))", // Center the content
				width: "100vw",
			}}
		>
			{(inspectedFeature ? ["left", "right"] : ["left"]).map(
				(columnSide: string) => (
					<FeatureColumn
						key={columnSide}
						inspectFeature={inspectFeature}
						inspectedFeature={inspectedFeature}
						columnSide={columnSide as "left" | "right"}
					/>
				)
			)}
		</div>
	);
}
