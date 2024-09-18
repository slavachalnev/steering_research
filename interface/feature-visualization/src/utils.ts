import { ProcessedFeaturesType } from "./types";

export const CONTAINER_WIDTH = "675px";
export const BORDER_RADIUS = "4px";
export const FONT_SIZE = "1rem";
export const PADDING = "10px";

export const getBaseUrl = () => {
	return process.env.NODE_ENV === "development"
		? "http://localhost:5000"
		: "https://steering-explorer-server.vercel.app";
};

export const processText = async (text: string) => {
	console.log("Submitted passage:", text);
	try {
		const url = `${getBaseUrl()}/get_unique_acts?text=${text}`;
		const response = await fetch(url, {
			method: "GET",
			headers: {
				"Content-Type": "application/json",
			},
		});
		const data = await response.json();

		return data;
	} catch (error) {
		console.error("Error processing text:", error);
		return null;
	}
};

export const getActivations = async (features: number[]) => {
	console.log(features);

	try {
		const url = `${getBaseUrl()}/get_feature?features=${features.join(",")}`;
		const response = await fetch(url, {
			method: "GET",
			headers: {
				"Content-Type": "application/json",
			},
		});
		const data = await response.json();
		const dataWithId: ProcessedFeaturesType[] = data.map((d: any) => {
			return Object.assign(d, {
				id: crypto.randomUUID(),
			});
		});
		console.log(dataWithId);
		return dataWithId;
	} catch (error) {
		console.error("Error getting activations:", error);
		return [];
	}
};
