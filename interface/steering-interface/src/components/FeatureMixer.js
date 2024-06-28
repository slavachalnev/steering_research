import React, { useState, useRef, useEffect } from "react";
import { steeringData } from "../utils/api";

// Function to determine if a color is light or dark
const getTextColor = (bgColor) => {
	const color = bgColor.charAt(0) === "#" ? bgColor.substring(1, 7) : bgColor;
	const r = parseInt(color.substring(0, 2), 16);
	const g = parseInt(color.substring(2, 4), 16);
	const b = parseInt(color.substring(4, 6), 16);
	const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
	return luminance > 186 ? "#000000" : "#FFFFFF";
};

export const MixturePreview = ({ features, setFeatures, style }) => {
	const [label, setLabel] = useState("");

	return (
		<div
			className="total-mixture"
			style={style}
			onDoubleClick={() => {
				if (setFeatures)
					setFeatures(features.map((feature) => ({ ...feature, value: 0 })));
			}}
			title={setFeatures ? "Double-click to reset all feature values to 0" : ""}
		>
			{Array.isArray(features) &&
				features.map((feature) => (
					<div
						key={feature.id}
						className="mixture-fill"
						onMouseEnter={() => setLabel(steeringData[feature.id])}
						onMouseLeave={() => setLabel("")}
						onClick={(ev) => {
							window.open(
								`https://www.neuronpedia.org/gemma-2b/6-res-jb/${feature.id}`,
								"_blank"
							);
						}}
						style={{
							minWidth: `${feature.value}%`,
							backgroundColor: feature.color,
							zIndex: 1, // Ensure this is viewed on top of mixture-fill-danger
						}}
					></div>
				))}
			<div
				className="mixture-fill-danger"
				style={{
					width: `20%`,
					backgroundColor:
						features.reduce((sum, feature) => sum + feature.value, 0) >= 80
							? "rgba(255, 0, 0, 0.2)"
							: "rgba(50, 50, 50, 0.2)",
				}}
			></div>
			<div className="mixture-label">{label}</div>
		</div>
	);
};

const FeatureMixer = ({ features, setFeatures }) => {
	const containerRef = useRef(null);
	const [containerWidth, setContainerWidth] = useState(0);

	useEffect(() => {
		if (containerRef.current) {
			setContainerWidth(containerRef.current.offsetWidth);
		}
	}, [features]);

	const handleMouseDown = (id, e) => {
		if (!containerRef.current) {
			console.error("containerRef is null");
			return;
		}
		const containerWidth = containerRef.current.offsetWidth;
		const startX = e.clientX;
		const startValue = features.find((feature) => feature.id === id).value;
		const totalValue = features.reduce(
			(sum, feature) => sum + feature.value,
			0
		);

		const featureDiv = e.target.closest(".feature-bar");
		if (featureDiv) {
			// Hide the mouse cursor and give the feature div a temporary border
			featureDiv.style.border = "1px dashed grey";
			featureDiv.style.marginTop = "-2px";
			featureDiv.style.marginLeft = "-2px";
		}

		const handleMouseMove = (e) => {
			const newX = e.clientX;
			const delta = ((newX - startX) / containerWidth) * 100;
			let newValue = Math.min(100, Math.max(0, startValue + delta));
			let newTotalValue = totalValue - startValue + newValue;

			if (newTotalValue > 100) {
				newValue -= newTotalValue - 100;
				newTotalValue = 100;
			}

			setFeatures((prevFeatures) =>
				prevFeatures.map((feature) =>
					feature.id === id ? { ...feature, value: newValue } : feature
				)
			);
		};

		const handleMouseUp = () => {
			window.removeEventListener("mousemove", handleMouseMove);
			window.removeEventListener("mouseup", handleMouseUp);
			document.body.style.userSelect = "";
			if (featureDiv) {
				// Restore the mouse cursor and remove the temporary border
				featureDiv.style.border = "";
				featureDiv.style.marginTop = "";
				featureDiv.style.marginLeft = "";
			}
		};

		window.addEventListener("mousemove", handleMouseMove);
		window.addEventListener("mouseup", handleMouseUp);

		// Disable user-select on mouse down
		document.body.style.userSelect = "none";

		// Set initial value on mouse down
		handleMouseMove(e);
	};

	return (
		<div className="feature-mixer" ref={containerRef}>
			{/* <div className="total-mixture">
				{Array.isArray(features) &&
					features.map((feature) => (
						<div
							key={feature.id}
							className="mixture-fill"
							style={{
								minWidth: `${feature.value}%`,
								backgroundColor: feature.color,
								zIndex: 1, // Ensure this is viewed on top of mixture-fill-danger
							}}
						></div>
					))}
				<div
					className="mixture-fill-danger"
					style={{
						width: `20%`,
						backgroundColor:
							features.reduce((sum, feature) => sum + feature.value, 0) >= 80
								? "rgba(255, 0, 0, 0.2)"
								: "rgba(50, 50, 50, 0.2)",
					}}
				></div>
			</div> */}
			<MixturePreview features={features} setFeatures={setFeatures} />
			{/* <button
				className="button"
				style={{
					marginLeft: "auto",
					marginBottom: "16px",
					marginRight: "0px",
				}}
				onClick={() =>
					setFeatures(features.map((feature) => ({ ...feature, value: 0 })))
				}
			>
				Reset Features
			</button> */}
			{Array.isArray(features) &&
				features.map((feature) => (
					<div key={feature.id} className="feature">
						<div
							className="feature-bar"
							onMouseDown={(e) => handleMouseDown(feature.id, e)}
							onDoubleClick={(e) => {
								setFeatures(
									features.map((currFeature) => {
										if (currFeature.id === feature.id) {
											return { ...currFeature, value: 0 };
										}
										return currFeature;
									})
								);
							}}
						>
							<div
								className="feature-fill"
								style={{
									width: `${feature.value}%`,
									backgroundColor: feature.color,
									color:
										feature.value > 2
											? getTextColor(feature.color)
											: getTextColor("#e0e0e0"),
								}}
							>
								{steeringData[feature.id]}
							</div>
							<div
								className="reverse-fill"
								style={{
									width: `${features.reduce(
										(sum, f) => (f.id !== feature.id ? sum + f.value : sum),
										0
									)}%`,
									backgroundColor:
										features.reduce((sum, feature) => sum + feature.value, 0) >=
										80
											? "rgba(255, 0, 0, 0.2)"
											: "rgba(50, 50, 50, 0.2)",
									position: "absolute",
									right: 0,
									top: 0,
									height: "100%",
									zIndex: 0,
								}}
							></div>
						</div>
					</div>
				))}
		</div>
	);
};

export default FeatureMixer;
