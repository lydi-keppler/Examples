-- Example SQL tasks
-- Tasks provided weekly by https://www.analystbuilder.com/questions
-- I use free datasets from Kaggle to do the task
-- I manipulate the data first in order to do the task, if needed
-- e.g., the task is to separate full_name into first_name and last_name, but it's already separate
-- > I will combine the names first, then separate 

-- ~~~~~~~~~~~~~~~~~~~~~~~~~~
-- Script BY Lydi Keppler; last edited Jan, 30, 2026
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~

-- 1. Addresses
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~
-- You are given a database containing customer addresses.
-- Write a query to break out the address column into separate columns for street, city, country.
-- Note: Some addresses may have additional unit or suite information 
-- (e.g., "Suite 5A" or "Unit B"), which should not be included as part of the street.
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~

-- Check
select * 
from student_name_and_address;
-- > No additional suite or unit info and the fields are already separate columns

-- New staging table that combines street, city, country into one column 
CREATE TABLE `students` (
    `student_name` TEXT,
    `full_address` TEXT
)  ENGINE=INNODB DEFAULT CHARSET=UTF8MB4 COLLATE = UTF8MB4_0900_AI_CI;

INSERT students
SELECT student_name, concat(street_address, ', ', city, ', ', country) as full_address 
FROM student_name_and_address;

-- Manually add suite or unit info for some addresses (randomly) 
UPDATE students
SET 
    full_address = '227 Petra Lodge, Unit B, North Alberta, Panama'
where student_name = 'Hugo O''Reilly'; 

UPDATE students
SET 
    full_address = '1496 N 5th Street, Suite 5A, Hickleberg, Democratic People''s Republic of Korea'
where student_name = 'Miss Louise Dibbert-Wiegand';

UPDATE students
SET 
    full_address = '204 The Grove, Suite 5D, Anderson, Nauru'
where student_name = 'Alice Kris IV';

UPDATE students
SET 
    full_address = '990 Raymond Ridges, Unit 3F, Strosinbury, Greece'
where student_name = 'Samantha Tremblay';

-- Find where we have 'unit' or 'suite' in the address
select *
from students
where full_address like '%unit %'
or full_address like '%suite %';

-- Separate addresses where we have 'unit' or 'suite' in the address
CREATE TEMPORARY TABLE WithUnit
select student_name as 'Name', full_address as 'Full address', 
SUBSTRING_INDEX(full_address, ",", 1) as Street, 
SUBSTRING_INDEX(SUBSTRING_INDEX(full_address, ',', 2), ',', -1) as Unit, 
SUBSTRING_INDEX(SUBSTRING_INDEX(full_address, ',', -2), ',', 1) as City, 
SUBSTRING_INDEX(full_address, ",", -1) as Country
from students
where full_address like '%unit %'
or full_address like '%suite %';

-- Separate addresses where we don't have 'unit' or 'suite' in the address
CREATE TEMPORARY TABLE WithoutUnit
select student_name as 'Name', full_address as 'Full address', 
SUBSTRING_INDEX(full_address, ",", 1) as Street, 
'N/A' as Unit, 
SUBSTRING_INDEX(SUBSTRING_INDEX(full_address, ',', -2), ',', 1) as City, 
SUBSTRING_INDEX(full_address, ",", -1) as Country
from students
where not full_address like '%unit %'
or full_address like '%suite %';

-- Combine and order by name
SELECT *
FROM WithUnit
UNION
SELECT *
FROM WithoutUnit
order by `Name`;
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~ 

