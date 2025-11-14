

CREATE TABLE IF NOT EXISTS mysql.Customers (
    customer_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100),
    phone VARCHAR(20),
    email VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS mysql.Vehicles (
    vehicle_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT,
    make VARCHAR(50),
    model VARCHAR(50),
    year INT,
    plate_number VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS mysql.Service_Records (
    service_id INT PRIMARY KEY AUTO_INCREMENT,
    vehicle_id INT,
    service_date DATE,
    description VARCHAR(255),
    cost DECIMAL(10,2)
);

-- Insert data
INSERT INTO mysql.Customers (name, phone, email) VALUES
('John Doe', '555-1234', 'john@example.com'),
('Sarah Miller', '555-5678', 'sarah@example.com'),
('David Smith', '555-9999', 'david@example.com');

INSERT INTO mysql.Vehicles (customer_id, make, model, year, plate_number) VALUES
(1, 'Toyota', 'Camry', 2018, 'ABC123'),
(2, 'Honda', 'Civic', 2020, 'XYZ789'),
(3, 'Ford', 'F-150', 2019, 'TRK456');

INSERT INTO mysql.Service_Records (vehicle_id, service_date, description, cost) VALUES
(1, CURDATE() - INTERVAL 10 DAY, 'Oil Change', 59.99),
(3, CURDATE() - INTERVAL 5 DAY, 'Engine Tune-Up', 250.00);

-- Print tables
SELECT * FROM mysql.Customers;
SELECT * FROM mysql.Vehicles;
SELECT * FROM mysql.Service_Records;

-- Query: Vehicles serviced in last 30 days
SELECT 
    v.vehicle_id,
    v.make,
    v.model,
    v.plate_number,
    sr.service_date,
    sr.description,
    c.name AS customer_name
FROM mysql.Service_Records sr
JOIN mysql.Vehicles v ON sr.vehicle_id = v.vehicle_id
JOIN mysql.Customers c ON v.customer_id = c.customer_id
WHERE sr.service_date >= CURDATE() - INTERVAL 30 DAY;
