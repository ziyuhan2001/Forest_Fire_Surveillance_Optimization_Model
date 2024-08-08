# Optimizing the Surveillance Path of Sensor-Equipped Drones for Early Wildfire Detection in the State of Arizona

## Introduction 
As shown throughout history, wildfires are a critical factor that emphasizes our climate is unsteady and destructive to life and the environment, yielding economic costs between $394 billion to $893 billion annually in the United States [1]. Wildfires present a severe societal concern not only due to the significant loss of property involved and its intense characteristics but also its impact on air quality, crops, resources, transportation, and the livelihood of animals and local populations. Posing a serious threat to the natural balance of ecosystems and global climate change, wildfires spread rapidly and are often abrupt, challenging to control and regulate, and difficult to initially locate and recognize. For wildfires, the likelihood of occurrence and corresponding strength depend on several factors, including weather conditions, vegetation characteristics, wind speed, precipitation, and topography [2]. In Arizona particularly, the extreme temperatures and dry conditions yield highly flammable plants and crops, causing many regions to be susceptible to wildfires. This makes Arizona the fourth most at-risk state for wildfires in the United States in 2023, based on the number of properties facing extreme risk [3]. Considering such factors, Arizona is an appropriate state candidate for wildfire analysis and the development of an effective surveillance plan for early wildfire detection. In addition to its devastating effects supplemented by land and weather conditions, wildfires can be challenging to detect within a certain time period before it reaches its deadly, uncontrollable climax. Wildfires are fundamentally characterized by distinguishable features, particularly flames and smoke. However, as flames of wildfires are typically hidden at the initial stage, smoke is the only observable attribute in detecting early wildfires [4]. Therefore, the development of an early wildfire surveillance model to promptly detect wildfires in the state of Arizona and effectively mitigate the corresponding widespread devastation on local communities and environmental habitats is of upmost importance. 

## Problem Description 
Two primary factors must be addressed in the case of a wildfire: the time elapsed between the fire’s ignition and the arrival of firefighters and the assessment of the fire’s magnitude to determine the necessity of an emergency intervention. Firstly, this response time must be minimized to ensure the fire can be effectively contained and controlled. Secondly, as manual wildfire assessment is expensive, dangerous, and challenging due to limited visibility, more effective firefighting and wildfire evaluation strategies that consider such factors must be developed and adopted. As such, having both an accurate, reliable, and safe wildfire surveillance system is essential to minimize catastrophic losses and the scale of wildfire destruction. Among various preventive measures, remote sensing-based early detection system is one of the fundamental methods for wildfire surveillance. These systems can quickly analyze and pinpoint potential wildfires, enabling rapid stabilization practices without exposing humans to hazardous conditions. Traditionally, satellite imagery [5] and wireless sensor networks [6] have been used for wildfire detection and risk assessment. However, both types of systems have their limitations. As satellite imagery has limited resolution, the collected data are often averaged over pixels, making it difficult to precisely pinpoint a fire’s location and size [7]. In addition, smoke emitted from the fires and the extensive distance between such satellites and the wildfires can significantly affect its effectiveness and its detection response time. For wireless sensor networks, as such sensors must be installed in forests and deployed beforehand, their coverage and functions are contingent upon the level of investment in their acquisition and setup [8]. Furthermore, the sensors are vulnerable to malfunctions or destruction during wildfires, yielding additional costs for maintenance or replacement [8]. Due to the limitations of conventional systems, sensor-equipped drones have emerged as a more practical and economic solution for early wildfire detection. These drones are capable of flying over designated areas to collect relevant data in real-time using their advanced sensors. As such, their enhanced maneuverability, autonomy, ease of deployment, and cost-effectiveness make them a highly advantageous technology in this field.  

## Objective and Decisions 
Developing an effective and efficient wildfire surveillance model requires the consideration and balancing of various factors. As such, the objective of this project is to develop a linear programming model to determine the optimal surveillance plan of sensor-equipped drones for early wildfire detection, focusing specifically on the state of Arizona. Given a set of drones and nodes evenly distributed throughout the state to represent the possible flight locations for drones, the developed model will determine the optimal surveillance route plan and optimal recharging amount during each time period for each drone. The model focuses on optimizing several key factors: it maximizes the total average wildfire risk coverage, the total area covered, and the total distance traveled by drones, while simultaneously minimizing the recharging costs throughout all time periods. These factors are evaluated within specified constraints, including limited drone availability and strict time limitations. Drone limitations were imposed on the model to account for budgetary constraints, while time limitations were imposed to ensure timely wildfire detection and response, preventing the wildfire from escalating beyond control. By accounting for these these factors, the proposed surveillance model aims to significantly mitigate the devastating impacts of wildfires, protect natural environments and properties, and facilitate quicker, safer wildfire management operations. 

## Literature Review 
A thorough literature review was conducted on several scholarly publications regarding the development of a wildfire surveillance plan through the use of drones and other similar methods. In recent years, unmanned aerial vehicles equipped with cameras and sensors have been proposed as a potential solution for early wildfire detection as shown in paper by  Martinez-de Dios et al. [14] and Merino et al. [15]. However, their wildfire surveillance models rely heavily on the use of image processing and geo-location techniques to obtain wildfire data instead of focusing on drone coverage and surveillance. A notable exception is the publication from Yuan et al., which provides a detailed review on the use of unmanned aerial vehicles for forest fire monitoring and detection [16]. Apart from the usage of unmanned aerial vehicles in the case of wildfire surveillance, multiple optimization-based frameworks have been proposed to determine the optimal location of lookout towers such as Fernandes et. al’s research paper [17]. Similar to the proposed surveillance model in this study, Fernandes et. al present an optimization model to calculate the optimal location and minimum number of lookout towers required to monitor a given forest area [17].  

## Assumptions 
To simplify the model, several assumptions are made. These assumptions can be broken down into the following three categories: drone settings, battery properties, recharging capabilities, and satellite characteristics. 

### Drone Settings 
The proposed wildfire surveillance model utilizes SmokeD PRO firefight drones. According to SmokeD Systems, these drones are equipped with sensitive optical sensors in their cameras, enabling them to detect initial signs of fire and smoke from considerable distances [9]. The drones have the following capabilities: 

- They can ascend to altitude as high as 80 meters. 

- They are capable of monitoring expansive areas up to 1000 square kilometers, with a wildfire detection radius of approximately 17840 meters. 

- They ensure 24/7, automatic fire detection and have an advanced image stabilization system. 

- They are designed for resilience, remaining stable and steady in windy conditions. 

In this study, the drones’ hovering altitude is excluded from the surveillance model, under the assumption that they will maintain an appropriate elevation to monitor regions without being affected by the fires. To expediate model results, two adjustments have been implemented in the baseline model: (1) each drone’s wildfire detection radius has been increased from 17840 meters to 45000 meters; (2) the maximum traveling distance between nodes during consecutive time periods for each drone is 85000 meters. 

### Battery Properties 
In the surveillance model, each drone is equipped with a battery specifically designed to define and regulate its operational range, ensuring realistic monitoring capabilities across designated areas. At the beginning of the model, the drones’ battery levels are fully charged to 100 units, the maximum capacity allowed. This upper bound prevents overcharging and ensures that the drones operate within realistic conditions, acknowledging that drones must have finite battery limits. With a lower limit of 0 units, the battery levels can range anywhere from 0 to 100 units, accommodating both integer and non-integer values to provide precise measurements of the remaining battery power. Each battery is configured to decrease linearly in power based on the distance traveled by the corresponding drone. This linearity assumption in the battery level consumption is critical to the operational planning of the drones as it allows for precise calculations in the battery level at any given time before recharging is necessary. As such, this battery model guarantees that drone movements are aligned with their remaining battery level, optimizing area coverage and ensuring the efficient use of battery power during surveillance operations. 

### Recharging Capabilities 
To accommodate the battery characteristics of the drones, each consists of a recharging system that permits a maximum recharging amount of 20 units at any of the designated locations provided during each time period. This upper bound limit is designed to reflect realistic conditions where extensive or full recharges may not be feasible due to time and resource limitations. The minimum recharge amount is 0 units, enabling situations where a drone may not need to recharge during certain time periods. At each time period, the recharging amount can take on any continuous value from 0 to 20 units, providing precise control over the battery management of drones to satisfy varying operational demands. Recharging consists of a time cost, with each unit of recharging incurring a cost of 0.005 units, emphasizing the realistic balance between maintaining operational efficiency and preventing unnecessary or excessive recharging. In addition, drones are capable of performing wildfire surveillance and recharging simultaneously at a designated location within the same time period. This capability assumes that drones maintain high efficiencies, allowing them to provide continuous surveillance coverage without the need for downtime exclusively for recharging in between time periods. 

### Satellite Features 
Despite several limitations associated with using satellites for wildfire surveillance, their capability to offer global visualizations can be advantageous in wildfire suppression and resource allocation, especially when resources such as firefighters are in high competition and demand. Given the restrictions on drone availability and the critical time constraints for early wildfire detection, employing satellites for wildfire surveillance under specific conditions can be a reasonable strategy. This study operates under the assumption that a satellite monitors a circular area with a 120000-meter radius at the center of Arizona. Given the defined limitations, the satellite will only be utilized if the average wildfire risk value in its coverage area falls below a sufficient threshold, signifying a low probability of wildfire occurrence in that region. 

## References
[1] Nilsen, Ella. “Wildfires Are Dealing a Massive Blow to Us Real Estate and Homeownership, Congressional Report Finds.” CNN, Cable News Network, 16 Oct. 2023, www.cnn.com/2023/10/16/us/wildfire-cost-us-economy-congressional-report-climate/index.html. 
[2] Pishahang, Mohammad, et al. “MCDM-Based Wildfire Risk Assessment: A Case Study on the State of Arizona.” MDPI, Multidisciplinary Digital Publishing Institute, 24 Nov. 2023, www.mdpi.com/2571-6255/6/12/449. 
[3] Martin, Shannon. “2023 U.S. Wildfire Statistics.” Bankrate, 2 Oct. 2023, www.bankrate.com/insurance/homeowners-insurance/wildfire-statistics/#wildfire-statistics-by-state. 
[4] Optimal Placement and Intelligent Smoke Detection Algorithm for Wildfire-Monitoring Cameras | IEEE Journals & Magazine | IEEE Xplore, ieeexplore.ieee.org/document/9068226/. Accessed 30 Apr. 2024.  
[5] Automatic Fire Perimeter Determination Using Modis Hotspots Information | IEEE Conference Publication | IEEE Xplore, ieeexplore.ieee.org/document/7870928. Accessed 30 Apr. 2024. 
[6] Haifeng Lin, et al. “A Fuzzy Inference and Big Data Analysis Algorithm for the Prediction of Forest Fire Based on Rechargeable Wireless Sensor Networks.” Sustainable Computing: Informatics and Systems, Elsevier, 20 May 2017, www.sciencedirect.com/science/article/abs/pii/S2210537917300288?via%3Dihub%5C. 
[7] “NASA Tracks Wildfires from above to Aid Firefighters Below.” NASA, NASA, 26 July 2023, www.nasa.gov/missions/aqua/nasa-tracks-wildfires-from-above-to-aid-firefighters-below/. 
[8] Akhloufi, Moulay A., et al. “Unmanned Aerial Vehicles for Wildland Fires: Sensing, Perception, Cooperation and Assistance.” MDPI, Multidisciplinary Digital Publishing Institute, 22 Feb. 2021, www.mdpi.com/2504-446X/5/1/15. 
[9] “Wildfire Drones - Drone Detection System for Firefighting.” SmokeD, 22 Sept. 2023, smokedsystem.com/drones/. 
[10] Bureau, US Census. “Glossary.” Census.Gov, 11 Apr. 2022, www.census.gov/programs-surveys/geography/about/glossary.html#par_textimage_13. 
[11] Bureau, US Census. “Tiger/Line Shapefiles.” Census.Gov, 9 Jan. 2024, www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html. 
[12] Bureau, US Census. “What We Do.” Census.Gov, 13 Dec. 2023, www.census.gov/about/what.html. 
[13] “Map: National Risk Index.” Map | National Risk Index, hazards.fema.gov/nri/map. Accessed 30 Apr. 2024. 
[14] Martínez-de Dios, José Ramiro, et al. “Automatic Forest-Fire Measuring Using Ground Stations and Unmanned Aerial Systems.” MDPI, Molecular Diversity Preservation International, 16 June 2011, www.mdpi.com/1424-8220/11/6/6328.  
[15] Merino, Luis, et al. “An Unmanned Aircraft System for Automatic Forest Fire Monitoring and Measurement - Journal of Intelligent & Robotic Systems.” SpringerLink, Springer Netherlands, 16 Aug. 2011, link.springer.com/article/10.1007/s10846-011-9560-x. 
[16] (PDF) A Survey on Technologies for Automatic Forest Fire Monitoring, Detection and Fighting Using Uavs and Remote Sensing Techniques, www.researchgate.net/publication/273912533_A_Survey_on_Technologies_for_Automatic_Forest_Fire_Monitoring_Detection_and_Fighting_Using_UAVs_and_Remote_Sensing_Techniques. Accessed 30 Apr. 2024. 
[17] Optimisation of Location and Number of Lidar Apparatuses For ..., www.researchgate.net/publication/245054438_Optimisation_of_location_and_number_of_lidar_apparatuses_for_early_forest_fire_detection_in_hilly_terrain. Accessed 30 Apr. 2024.
