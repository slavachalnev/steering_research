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
			document.body.style.cursor = "none";
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
				document.body.style.cursor = "";
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

	useEffect(() => {
		console.log(features);
	}, [features]);

	return (
		<div className="feature-mixer" ref={containerRef}>
			<div className="total-mixture">
				{Array.isArray(features) &&
					features.map((feature) => (
						<div
							key={feature.id}
							className="mixture-fill"
							style={{
								minWidth: `${(feature.value / 100) * containerWidth}px`,
								backgroundColor: feature.color,
								zIndex: 1, // Ensure this is viewed on top of mixture-fill-danger
							}}
						></div>
					))}
				<div
					className="mixture-fill-danger"
					style={{
						width: `${(20 / 100) * containerWidth}px`,
					}}
				></div>
			</div>
			{Array.isArray(features) &&
				features.map((feature) => (
					<div key={feature.id} className="feature">
						<div
							className="feature-bar"
							onMouseDown={(e) => handleMouseDown(feature.id, e)}
						>
							<div
								className="feature-fill"
								style={{
									width: `${(feature.value / 100) * containerWidth}px`,
									backgroundColor: feature.color,
									color: getTextColor(feature.color),
								}}
							>
								{steeringData[feature.id]}
							</div>
						</div>
					</div>
				))}
			<button
				className="button"
				style={{
					marginLeft: "16px",
					marginBottom: "16px",
				}}
				onClick={() =>
					setFeatures(features.map((feature) => ({ ...feature, value: 0 })))
				}
			>
				Reset Features
			</button>
		</div>
	);
};

export default FeatureMixer;
