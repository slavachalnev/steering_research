import React, { useState, useRef, useEffect } from "react";

const FeatureMixer = () => {
	const [features, setFeatures] = useState([
		{ id: 1, value: 0, color: "orange" },
		{ id: 2, value: 60, color: "purple" },
		{ id: 3, value: 30, color: "blue" },
	]);

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
			featureDiv.style.border = "1px dashed red";
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
				featureDiv.style.marginTop = "0px";
				featureDiv.style.marginLeft = "0px";
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
								width: `${(feature.value / 100) * containerWidth}px`,
								backgroundColor: feature.color,
							}}
						></div>
					))}
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
								}}
							></div>
						</div>
					</div>
				))}
		</div>
	);
};

export default FeatureMixer;
