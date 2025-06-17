# KAIM_W03_AlphaCare_Insurance_Risk_Analysis (AlphaCare Insurance Solutions: Car Insurance Risk and Marketing Analytics)
AlphaCare Insurance Solutions (ACIS) is focused on advancing risk and predictive analytics for car insurance within South Africa. As a Marketing Analytics Engineer, your role involves analyzing historical insurance claim data to optimize marketing strategies, identify "low-risk" targets for premium adjustments, and attract new clients.

## Project Overview
This project focuses on analyzing historical car insurance claim data from South Africa to optimize AlphaCare Insurance Solutions' marketing strategy and identify "low-risk" client segments. The goal is to enable ACIS to reduce premiums for low-risk targets, thereby attracting new clients.

## Data
The analysis utilizes historical insurance claim data from February 2014 to August 2015, provided as `MachineLearningRating_v3.txt`. This dataset includes detailed information on insurance policies, client demographics and location, vehicle specifications, plan details, and payment/claim information.

**Key Data Columns:**
* **Policy & Transaction:** `UnderwrittenCoverID`, `PolicyID`, `TransactionMonth`
* **Client:** `IsVATRegistered`, `Citizenship`, `LegalType`, `Title`, `Language`, `Bank`, `AccountType`, `MaritalStatus`, `Gender`
* **Location:** `Country`, `Province`, `PostalCode`, `MainCrestaZone`, `SubCrestaZone`
* **Car:** `ItemType`, `Mmcode`, `VehicleType`, `RegistrationYear`, `Make`, `Model`, `Cylinders`, `Cubiccapacity`, `Kilowatts`, `Bodytype`, `NumberOfDoors`, `VehicleIntroDate`, `CustomValueEstimate`, `AlarmImmobiliser`, `TrackingDevice`, `CapitalOutstanding`, `NewVehicle`, `WrittenOff`, `Rebuilt`, `Converted`, `CrossBorder`, `NumberOfVehiclesInFleet`
* **Plan:** `SumInsured`, `TermFrequency`, `CalculatedPremiumPerTerm`, `ExcessSelected`, `CoverCategory`, `CoverType`, `CoverGroup`, `Section`, `Product`, `StatutoryClass`, `StatutoryRiskType`
* **Financial:** `TotalPremium`, `TotalClaims`

## Project Structure
The repository is structured as follows:
