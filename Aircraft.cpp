aaabb
#include "Aircraft.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <algorithm>

using namespace std;

void Aircraft::Set_Parameters_and_Data(string File_Path, string droneFilePath, int no_cols){
    FilePath = File_Path;
    TotalCols = no_cols;
    // Callsign column number
    callsign = 3;
    // Longitude column number
    longitude = 14;
    // Latitude column number
    latitude = 13;
    // Altitude column number 
    altitude = 7;
    // Groundspeed column number
    groundspeed = 8;
    // Onground Column number
    onground = 15;
    // Heading Column number
    track = 20;
    // Airport Destination column number
    airport_dest = 5;
    // Drone target runway column number
    runway = 23;
    // Aircraft Vertical Rate
    verticalRate = 21;


    CSVDataAircraft();
    CSVDataDroneAreas(droneFilePath);
    AircraftIndex();
}



void Aircraft::CSVDataAircraft(){
    ifstream csv;

    csv.open(FilePath);

    if (csv.fail()) {
        cout << "ERROR - Failed to open " << FilePath << endl;
    }

    string line;
    string element;
    
    // Seperating line by line and then element by element 
    while(getline(csv, line)){
        stringstream new_line(line);
        while (getline(new_line, element, ',')){
        AllData.push_back(element);
        }
    }
    csv.close();

    AllData_size = AllData.size();
}

void Aircraft::CSVDataDroneAreas(string File_Path){
    ifstream csv;

    csv.open(File_Path);

    if (csv.fail()) {
        cout << "ERROR - Failed to open " << File_Path << endl;
    }

    string line;
    string element;
    
    // Seperating line by line and then element by element 
    while(getline(csv, line)){
        stringstream new_line(line);
        while (getline(new_line, element, ',')){
        droneAreaData.push_back(element);
        }
    }
    csv.close();

    droneAreaData_size = droneAreaData.size();
}


void Aircraft::AircraftIndex(){
    for(int i = 0; i < AllData_size / TotalCols; ++i){
        if ((i+1)*TotalCols + callsign >= AllData_size) {
            // The next aircraft does not exist, so break out of the loop
            break;
        }
        if (AllData[i*TotalCols + callsign] != AllData[(i+1)*TotalCols + callsign]) {
            Aircraft_Index.push_back((i+1)*TotalCols);    
        }
    }
    Aircraft_Index_size = Aircraft_Index.size();
}


void Aircraft::SingleAircraft(int index){
    if (index == Aircraft_Index_size){
        for (int i = Aircraft_Index[index-1]; i < AllData_size; ++i){
        Single_Aircraft.push_back(AllData[i]);
        }
    }
    else{
        for (int i = Aircraft_Index[index]; i < Aircraft_Index[index+1]; ++i){
            Single_Aircraft.push_back(AllData[i]);
        }
    }
    Single_Aircraft_size = Single_Aircraft.size();
}

void Aircraft::ColumnSelect(int column_no, double* column_pointer){
  vector<double> column;
  
  for(int i = 0; i < Single_Aircraft_size / TotalCols; ++i){
    column.push_back(stod(Single_Aircraft[(i*TotalCols) + column_no]));
  }

  int column_size = column.size();

  for (int i = 0; i < column_size; ++i){ 
    column_pointer [i] = column[i];
  }
  
}


void Aircraft::Takeoff_Time(){
    takeoff_t = 0;
    for(int i = 0; i < Single_Aircraft_size / TotalCols; ++i){
        if(Single_Aircraft[i*TotalCols + onground] != Single_Aircraft[(i+1)*TotalCols + onground]){
            break;
        }
        takeoff_t += 1;
    }
}

void Aircraft::Arrive_Time(){
    arrive_t = 0;
    
    for(int i = 0; i < Single_Aircraft_size / TotalCols; ++i){
        if(stod(Single_Aircraft[i*TotalCols + altitude]) <= 275.0){
            break;
        }
        arrive_t += 1;
    }
    /*
    if(Single_Aircraft[0*TotalCols + airport_dest] == "EGKK"){
        for(int i = 0; i < Single_Aircraft_size / TotalCols; ++i){
            if(stod(Single_Aircraft[i*TotalCols + altitude]) <= 225.0){
                break;
            }
            arrive_t += 1;
        }
    }

    if(Single_Aircraft[0*TotalCols + airport_dest] == "EGLL"){
        for(int i = 0; i < Single_Aircraft_size / TotalCols; ++i){
            if(stod(Single_Aircraft[i*TotalCols + altitude]) <= 350.0){
                break;
            }
            arrive_t += 1;
        }
    }
    */
}

void Aircraft::DroneRunwayAreaIdentifier(int rows){
    int totalCols = droneAreaData_size / rows;
    for(int j = 1; j < totalCols; ++j){
        droneAreaData[j].erase(remove_if(droneAreaData[j].begin(), droneAreaData[j].end(), ::isspace), droneAreaData[j].end());
        if (droneTargetRunway == droneAreaData[j]){
            
            for(int i = 0; i < rows-2; ++i){
                droneAreaLat[i] = stod(droneAreaData[(i+2) * totalCols + j]);
                droneAreaLon[i] = stod(droneAreaData[(i+2) * totalCols + j + 1]);
            }
            break;
        }
    }
}

void Aircraft::Vector_Allocation(int index_input, int droneAreaRows){
    SingleAircraft(index_input);
    Takeoff_Time();
    Arrive_Time();
    Vector_length = Single_Aircraft_size/TotalCols;

    longitude_vector = new double[Vector_length];
    latitude_vector = new double[Vector_length];
    altitude_vector = new double[Vector_length];
    groundspeed_vector = new double[Vector_length];
    track_vector = new double[Vector_length];
    verticalRate_vector = new double[Vector_length];

    droneAreaLat = new double[droneAreaRows];
    droneAreaLon = new double[droneAreaRows];

    ColumnSelect(longitude, longitude_vector);
    ColumnSelect(latitude, latitude_vector);
    ColumnSelect(altitude, altitude_vector);
    ColumnSelect(groundspeed, groundspeed_vector);
    ColumnSelect(track, track_vector);
    ColumnSelect(verticalRate, verticalRate_vector);

    // Obtaining the runway name from the aircraft data.
    droneTargetRunway = Single_Aircraft[runway];
    droneTargetRunway.erase(remove_if(droneTargetRunway.begin(), droneTargetRunway.end(), ::isspace), droneTargetRunway.end()); 
    DroneRunwayAreaIdentifier(droneAreaRows);
}

void Aircraft::Deallocation(){
    delete[] longitude_vector;
    delete[] latitude_vector;
    delete[] altitude_vector;
    delete[] groundspeed_vector;
    delete[] track_vector;
    delete[] verticalRate_vector;
    delete[] droneAreaLat;
    delete[] droneAreaLon;
    Single_Aircraft.clear();
}


void Aircraft::PrintAircraft(){
  cout.precision(4);
  for (int i = 0; i < Single_Aircraft_size / TotalCols ; ++i){
    for (int j = 0; j < TotalCols; ++j){
      cout << setw(4) << Single_Aircraft[i*TotalCols + j] << " ";
    }
    cout << endl;
  }
}
