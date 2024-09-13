import React from "react";
import FeatureCard from "./FeatureCard";

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
	onMagnify,
	removeFeature,
}: {
	processedFeatures: ProcessedFeaturesType[];
	onMagnify: (id: string) => void;
	removeFeature: (id: string) => void;
}) => {
	return (
		<div
			style={{
				height: "80vh",
				display: "flex",
				flexDirection: "column",
				overflow: "hidden",
			}}
		>
			<div
				style={{
					display: "flex",
					flexDirection: "column",
					transition: "width 0.3s",
					flexGrow: 1,
					overflowY: "auto",
				}}
			>
				<div style={{ display: "flex", flexDirection: "column", gap: ".5px" }}>
					{processedFeatures.map(
						(feature: ProcessedFeaturesType, i: number) => {
							const maxAct = feature.feature_results.reduce((max, current) =>
								current.maxValue > max.maxValue ? current : max
							).binMax;
							return (
								<React.Fragment key={feature.id + i}>
									<FeatureCard
										feature={feature.feature}
										featureId={feature.id}
										onDelete={removeFeature}
										onMagnify={onMagnify}
										activations={feature.feature_results}
										maxAct={maxAct}
									/>
									<div style={{ height: "10px", width: "100%" }} />
								</React.Fragment>
							);
						}
					)}
				</div>
			</div>
		</div>
	);
};

export default FeatureColumn;
