
CREATE TABLE childs (
    id VARCHAR(20) PRIMARY KEY,              
    prenom VARCHAR(50) NOT NULL,            
    age INT NOT NULL,                       
    ville_residence VARCHAR(100) NOT NULL,   
    structure_accueil VARCHAR(100) NOT NULL 
);

CREATE TABLE journeys (
    id VARCHAR(20) PRIMARY KEY,             
    date DATE NOT NULL,                    
    enfant_id VARCHAR(20) NOT NULL,        
    type_demande VARCHAR(100) NOT NULL,       
    heure_depart TIME NOT NULL,              
    lieu_depart VARCHAR(100) NOT NULL,     
    lieu_arrivee VARCHAR(100) NOT NULL,     
    distance_km DECIMAL(6,2) NOT NULL,    
    duree_estimee INT NOT NULL,             
    cout_taxi_estime DECIMAL(8,2) NOT NULL  
);

CREATE TABLE drivers (
    driver_id SERIAL PRIMARY KEY,
    prenom VARCHAR(50) NOT NULL,                       
    secteur_attribue VARCHAR(100) NOT NULL,        
    vehicule VARCHAR(100) NOT NULL,                  
    cout_kilometrique DECIMAL(6,3) NOT NULL,        
    capacite_passagers INT NOT NULL,              
    lieu_garage VARCHAR(100) NOT NULL             
);
