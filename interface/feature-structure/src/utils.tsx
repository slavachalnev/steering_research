export const MIN_CARD_HEIGHT = 100;

const getBaseUrl = () => {
	return process.env.NODE_ENV === "development"
		? "http://localhost:5000"
		: "https://steering-explorer-server.vercel.app";
};

export const fetchExpandedCluster = async (
	feature: number,
	cluster: number[],
	threshold: number = 0.5
) => {
	console.log(
		`fetching activations for feature: ${feature} and cluster: ${cluster}`
	);
	try {
		const response = await fetch(`${getBaseUrl()}/expand_cluster`, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({
				node_to_expand: feature,
				current_cluster: cluster,
				threshold: threshold,
			}),
		});
		if (!response.ok) {
			throw new Error("Network response was not ok");
		}
		const rawData = await response.json();
		return rawData;
	} catch (error) {
		console.error("There was a problem fetching activations:", error);
	}
};
